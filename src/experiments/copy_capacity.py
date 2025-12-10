import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.plastic_transformer import PlasticTransformerModel
from src.tasks.copying import CopyingTaskConfig, CopyingTaskDataset



#

def resolve_device(preferred: str) -> torch.device:
    """
    Determine which device (CPU/GPU) to use for training.
    
    Args:
        preferred: Either "auto" (auto-detect) or specific device ("cpu", "cuda", "mps")
    
    Returns:
        torch.device object representing the chosen device
    """
    # If user specified a device, use it (with validation)
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return device
    
    # Auto-detect: prefer MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    Ensures that runs with the same seed produce identical results.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy random number generator
    torch.manual_seed(seed)  # PyTorch CPU random number generator
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random number generators
    torch.backends.cudnn.deterministic = True  # Make CUDA operations deterministic
    torch.backends.cudnn.benchmark = False  # Disable CUDA auto-tuner for reproducibility


def summarise(values: List[float]) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of a list of values.
    
    Args:
        values: List of numeric values to summarize
    
    Returns:
        Tuple of (mean, std_dev)
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))  # ddof=0 for population std


def aggregate_metrics(finals: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple experimental runs.
    Computes mean and standard deviation for each metric.
    
    Args:
        finals: List of dictionaries containing final metrics from each run
    
    Returns:
        Dictionary with aggregated statistics (mean and std for each metric)
    """
    if not finals:
        return {}
    
    # Get all metric keys from the first run
    keys = finals[0].keys()
    aggregate: Dict[str, float] = {"num_runs": len(finals)}
    
    # For each metric, compute mean and std across all runs
    for key in keys:
        # Extract values for this metric from all runs (skip non-numeric values)
        vals = [run[key] for run in finals if isinstance(run.get(key), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std(ddof=0))
    return aggregate


@dataclass
class TrainingConfig:
    """Configuration parameters for training the model."""
    epochs: int = 5  # Number of training epochs
    lr: float = 1e-3  # Learning rate for optimizer
    weight_decay: float = 1e-4  # L2 regularization strength
    clip_norm: float = 5.0  # Maximum gradient norm (for gradient clipping)
    log_interval: int = 50  # How often to log training progress (in steps)


def train_epoch(
    model: PlasticTransformerModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    clip_norm: float,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, Dict[str, float]]:
    """
    Train the model for one epoch on the copying task.
    Processes sequences one time step at a time, accumulating loss across the entire sequence.
    
    Args:
        model: The plastic transformer model to train
        dataloader: DataLoader providing copying task sequences
        device: Device to run computations on
        loss_fn: Loss function (CrossEntropyLoss)
        clip_norm: Maximum gradient norm for gradient clipping
        optimizer: Optimizer for updating model parameters
    
    Returns:
        Tuple of (average_loss, diagnostics_dict)
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    total_steps = 0
    eta_values: List[float] = []  # Track plasticity learning rates
    plastic_values: List[float] = []  # Track plastic weight magnitudes
    progress = tqdm(dataloader, desc="train", leave=False)
    
    for inputs, targets in progress:
        # Remove batch dimension (batch_size=1 for this task)
        # Shape: (sequence_length,) containing token indices
        inputs = inputs.squeeze(0).long().to(device)
        targets = targets.squeeze(0).long().to(device)
        
        # Initialize plastic state for this trial (resets plastic weights to zero)
        state = model.init_state(device)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        
        # Process sequence one time step at a time
        for t in range(inputs.shape[0]):
            # Convert token index to one-hot vector (model expects continuous input)
            x_vec = F.one_hot(inputs[t], num_classes=model.output_dim).float().to(device)
            
            # Forward step: processes one token, updates plastic weights
            outputs = model.forward_step(x_vec, state)
            
            # Compute loss for this time step
            logits = outputs["logits"].unsqueeze(0)  # Add batch dimension for loss_fn
            step_target = targets[t].unsqueeze(0)
            step_loss = loss_fn(logits, step_target)
            loss = loss + step_loss  # Accumulate loss across all time steps
            
            # Track plasticity diagnostics
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        
        # Backpropagate through the entire sequence
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # Update model parameters (static weights, not plastic weights)
        optimizer.step()
        
        total_loss += loss.item()
        total_steps += inputs.shape[0]
        if total_steps > 0:
            progress.set_postfix(loss=loss.item() / inputs.shape[0])
    
    # Compute diagnostic statistics
    diag = {
        "eta_mean": summarise(eta_values)[0],
        "eta_std": summarise(eta_values)[1],
        "plastic_norm_mean": summarise(plastic_values)[0],
        "plastic_norm_std": summarise(plastic_values)[1],
    }
    return total_loss / max(total_steps, 1), diag


def evaluate(
    model: PlasticTransformerModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """
    Evaluate the model on validation/test data.
    Measures loss and recall accuracy (how well it remembers the sequence).
    
    Args:
        model: The plastic transformer model to evaluate
        dataloader: DataLoader providing copying task sequences
        device: Device to run computations on
        loss_fn: Loss function (CrossEntropyLoss)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()  # Set model to evaluation mode
    losses: List[float] = []
    correct_recall = 0  # Number of correct predictions during recall phase
    total_recall = 0  # Total number of recall phase predictions
    eta_values: List[float] = []
    plastic_values: List[float] = []
    
    # Gradient-based plasticity needs gradients even during evaluation
    # Other rules can run without gradients for efficiency
    context = torch.enable_grad if model.rule == "gradient" else torch.no_grad
    
    with context():
        for inputs, targets in dataloader:
            # Remove batch dimension
            inputs = inputs.squeeze(0).long().to(device)
            targets = targets.squeeze(0).long().to(device)
            
            # Initialize fresh plastic state for this trial
            state = model.init_state(device)
            trial_loss = 0.0
            
            # Process sequence one time step at a time
            for t in range(inputs.shape[0]):
                # Convert token index to one-hot vector
                x_vec = F.one_hot(inputs[t], num_classes=model.output_dim).float().to(device)
                
                # Forward step with plastic weight updates
                outputs = model.forward_step(x_vec, state)
                logits = outputs["logits"]
                step_target = targets[t].unsqueeze(0)
                trial_loss += loss_fn(logits.unsqueeze(0), step_target).item()
                
                # Track diagnostics
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
                
                # Only evaluate accuracy during the recall phase (second half of sequence)
                # This is when the model should be outputting the memorized sequence
                if t >= inputs.shape[0] // 2:
                    pred = logits.argmax().item()  # Get predicted token
                    if pred == targets[t].item():
                        correct_recall += 1
                    total_recall += 1
            
            losses.append(trial_loss / inputs.shape[0])
    
    # Compute aggregate metrics
    avg_loss = sum(losses) / max(len(losses), 1)
    recall_acc = correct_recall / max(total_recall, 1)  # Accuracy during recall phase
    eta_mean, eta_std = summarise(eta_values)
    plastic_mean, plastic_std = summarise(plastic_values)
    
    return {
        "loss": avg_loss,
        "recall_accuracy": recall_acc,  # This is the key metric!
        "eta_mean": eta_mean,
        "eta_std": eta_std,
        "plastic_norm_mean": plastic_mean,
        "plastic_norm_std": plastic_std,
    }


def execute_single_run(args: argparse.Namespace, device: torch.device, seed: int) -> Dict[str, List[Dict[str, float]]]:
    """
    Execute a single experimental run with a specific random seed.
    This includes dataset creation, model initialization, training, and evaluation.
    
    Args:
        args: Command-line arguments containing hyperparameters
        device: Device to run the experiment on
        seed: Random seed for this run
    
    Returns:
        Dictionary containing training history and final metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    history: List[Dict[str, float]] = []
    finals: List[Dict[str, float]] = []

    for seq_length in range(args.seq_low_length, args.seq_high_length + 1):
    
        # Create copying task configuration
        copying_cfg = CopyingTaskConfig(
            seq_length=seq_length,
            delay=args.delay,
            vocab_size=args.vocab_size,
            dataset_size=args.dataset_size,
        )
        
        # Create train and validation datasets (both randomly generated)
        train_dataset = CopyingTaskDataset(copying_cfg)
        val_dataset = CopyingTaskDataset(copying_cfg)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialize the plastic transformer model
        # input_dim and output_dim are both vocab_size (one-hot encoding)
        model = PlasticTransformerModel(
            input_dim=args.vocab_size,
            output_dim=args.vocab_size,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            aux_dim=args.aux_dim,
            rule=args.rule,  # "none", "hebbian", or "gradient"
            eta0=args.eta0,  # Base plasticity learning rate
            max_norm=args.max_norm,  # Maximum norm for plastic updates
        ).to(device)

        # Training configuration
        train_cfg = TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            clip_norm=args.clip_norm,
            log_interval=args.log_interval,
        )
        
        # Setup optimizer (AdamW with weight decay for regularization)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        
        # Loss function for predicting token indices
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(train_cfg.epochs):
            # Train for one epoch
            train_loss, train_diag = train_epoch(model, train_loader, device, loss_fn, train_cfg.clip_norm, optimizer)
            
            # Evaluate on validation set
            metrics = evaluate(model, val_loader, device, loss_fn)
            
            # Combine training and validation metrics
            metrics["seq_length"] = seq_length
            metrics["train_loss"] = train_loss
            metrics["train_eta_mean"] = train_diag["eta_mean"]
            metrics["train_eta_std"] = train_diag["eta_std"]
            metrics["train_plastic_norm_mean"] = train_diag["plastic_norm_mean"]
            metrics["train_plastic_norm_std"] = train_diag["plastic_norm_std"]
            metrics["epoch"] = epoch + 1
            history.append(metrics)
            
            # Print progress
            print(
                f"[seq_length={seq_length}],"
                f"[seed={seed}] Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={metrics['loss']:.4f}, recall_acc={metrics['recall_accuracy']:.4f}"
            )
            if metrics['recall_accuracy'] == 1.0:
                break
        finals.append(metrics)
    
    return {
        "history": history,  # Metrics for each epoch
        "final": finals,  # Final epoch metrics
        "config": asdict(copying_cfg),  # Task configuration
        "training": asdict(train_cfg),  # Training configuration
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    """
    Run the complete experiment with multiple random seeds.
    Executes multiple independent runs and aggregates the results.
    
    Args:
        args: Command-line arguments containing all hyperparameters
    
    Returns:
        Final metrics from the last run
    """
    # Determine which device to use
    device = resolve_device(args.device)
    
    # Generate list of seeds for multiple runs
    # e.g., if base_seed=123 and seeds=3, then seeds=[123, 124, 125]
    seeds = [args.base_seed + i for i in range(args.seeds)]
    
    run_summaries = []  # Store results from each run
    final_metrics: List[Dict[str, float]] = []  # Store final epoch metrics from each run
    task_config = None
    training_config = None

    # Execute multiple runs with different seeds
    for seed in seeds:
        summary = execute_single_run(args, device, seed)
        run_summaries.append({"seed": seed, "history": summary["history"], "final": summary["final"]})
        final_metrics.append(summary["final"])
        
        # Save config from first run (same for all runs)
        if task_config is None:
            task_config = summary["config"]
        if training_config is None:
            training_config = summary["training"]

    # Compute aggregate statistics across all runs (mean ± std)
    aggregate = aggregate_metrics(final_metrics)
    last_final = final_metrics[-1] if final_metrics else {}

    # Save results to JSON file if output path specified
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        payload = {
            "config": {
                "model": {
                    "rule": args.rule,
                    "model_dim": args.model_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "ffn_dim": args.ffn_dim,
                },
                "task": task_config,
                "training": training_config,
            },
            "runs": run_summaries,  # Individual run results
            "aggregate": aggregate,  # Aggregated statistics
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    
    return last_final


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the copying task experiment.
    
    Returns:
        Namespace object containing all parsed arguments
    """
    parser = argparse.ArgumentParser(description="Copying task experiment runner")
    
    # Plasticity rule selection
    parser.add_argument("--rule", choices=["none", "hebbian", "gradient"], default="hebbian",
                        help="Plasticity rule: 'none' (no plasticity), 'hebbian' (Hebbian learning), or 'gradient' (gradient-based)")
    
    # Task parameters
    parser.add_argument("--seq-low-length", type=int, default=1,
                        help="Minimum length of sequence to memorize and recall")
    parser.add_argument("--seq-high-length", type=int, default=20,
                        help="Maximum length of sequence to memorize and recall")
    parser.add_argument("--delay", type=int, default=5,
                        help="Number of blank steps between presentation and recall")
    parser.add_argument("--vocab-size", type=int, default=10,
                        help="Size of token vocabulary (includes blank and delimiter)")
    parser.add_argument("--dataset-size", type=int, default=500,
                        help="Number of randomly generated sequences per dataset")
    
    # Model architecture
    parser.add_argument("--model-dim", type=int, default=128,
                        help="Dimension of transformer hidden states")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of transformer blocks")
    parser.add_argument("--ffn-dim", type=int, default=256,
                        help="Dimension of feedforward network intermediate layer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--aux-dim", type=int, default=4,
                        help="Dimension of auxiliary output vector")
    
    # Plasticity parameters
    parser.add_argument("--eta0", type=float, default=0.2,
                        help="Base plasticity learning rate")
    parser.add_argument("--max-norm", type=float, default=1.0,
                        help="Maximum norm for plasticity updates (0 disables clipping)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization) coefficient")
    parser.add_argument("--clip-norm", type=float, default=5.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Number of steps between logging updates")
    
    # Experiment setup
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Device to run on: 'auto' (auto-detect), 'cpu', 'cuda' (NVIDIA GPU), or 'mps' (Apple Silicon)")
    parser.add_argument("--base-seed", type=int, default=123,
                        help="Base random seed (incremented for multiple runs)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of independent runs with different seeds")
    parser.add_argument("--output-path", type=str, default="",
                        help="Path to save results JSON file (empty = don't save)")
    
    return parser.parse_args()


if __name__ == "__main__":
    """
    Main entry point when running this script directly.
    Parses command-line arguments and executes the experiment.
    """
    args = parse_args()
    run_experiment(args)
