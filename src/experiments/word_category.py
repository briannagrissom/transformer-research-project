"""
Word category classification experiment.

This experiment tests a plastic transformer's ability to perform few-shot
classification on word categories (animals, clothing, furniture).
"""
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

from src.models.text_encoder import CharacterEncoder
from src.models.onehot_text_encoder import OneHotTextEncoder
from src.models.word2vec_encoder import SimpleSemanticEncoder, Word2VecEncoder
from src.models.plastic_transformer import PlasticTransformerModel
from src.tasks.word_category_classification import WordCategoryConfig, WordCategoryTask, ALL_WORDS, CATEGORIES


def resolve_device(preferred: str) -> torch.device:
    """Resolve the device to use for computation."""
    print(f"Resolving device (preferred: {preferred})...", flush=True)
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def summarise(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std of values."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_metrics(finals: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple runs."""
    print("Aggregating metrics across runs...", flush=True)
    if not finals:
        return {}
    aggregate: Dict[str, float] = {"num_runs": len(finals)}
    keys = finals[0].keys()
    for key in keys:
        vals = [run[key] for run in finals if isinstance(run.get(key), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std(ddof=0))
    return aggregate


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    episodes_per_epoch: int = 200
    val_episodes: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    clip_norm: float = 5.0


def build_support_vector(embedding: torch.Tensor, label_one_hot: torch.Tensor) -> torch.Tensor:
    """Build input vector for a support example."""
    return torch.cat([embedding, label_one_hot, torch.zeros(1, device=embedding.device)])


def build_query_vector(embedding: torch.Tensor, ways: int) -> torch.Tensor:
    """Build input vector for a query example."""
    return torch.cat(
        [
            embedding,
            torch.zeros(ways, device=embedding.device),
            torch.ones(1, device=embedding.device),
        ]
    )


def train_epoch(
    encoder: nn.Module,
    transformer: PlasticTransformerModel,
    task: WordCategoryTask,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: TrainingConfig,
    ways: int,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch (multiple episodes)."""
    encoder.train()
    transformer.train()
    total_loss = 0.0
    eta_values: List[float] = []
    plastic_values: List[float] = []
    
    for ep_idx in range(config.episodes_per_epoch):
        if ep_idx % 10 == 0:
            print(f"  Training episode {ep_idx+1}/{config.episodes_per_epoch}", flush=True)
        
        # Sample episode
        support_strings, support_labels, query_strings, query_labels, category_names = task.sample_episode()
        
        # Encode strings to embeddings
        support_embeddings = encoder(support_strings)
        query_embeddings = encoder(query_strings)
        
        # Convert labels to tensors
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)
        
        # Initialize transformer state
        state = transformer.init_state(device)
        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0, device=device)
        
        # Process support set (adapt plastic weights)
        for idx in range(support_embeddings.shape[0]):
            embedding = support_embeddings[idx]
            label = support_labels_tensor[idx]
            one_hot = F.one_hot(label, num_classes=ways).float()
            step_vec = build_support_vector(embedding, one_hot)
            outputs = transformer.forward_step(step_vec, state)
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        
        # Process query set (compute loss)
        for idx in range(query_embeddings.shape[0]):
            embedding = query_embeddings[idx]
            label = query_labels_tensor[idx]
            step_vec = build_query_vector(embedding, ways)
            outputs = transformer.forward_step(step_vec, state)
            logits = outputs["logits"].unsqueeze(0)
            episode_loss = episode_loss + loss_fn(logits, label.unsqueeze(0))
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        
        # Backpropagate and update weights
        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(transformer.parameters()),
            config.clip_norm
        )
        optimizer.step()
        total_loss += episode_loss.item() / max(query_embeddings.shape[0], 1)
    
    diag = {
        "eta_mean": summarise(eta_values)[0],
        "eta_std": summarise(eta_values)[1],
        "plastic_norm_mean": summarise(plastic_values)[0],
        "plastic_norm_std": summarise(plastic_values)[1],
    }
    return total_loss / max(config.episodes_per_epoch, 1), diag


def evaluate(
    encoder: nn.Module,
    transformer: PlasticTransformerModel,
    task: WordCategoryTask,
    device: torch.device,
    loss_fn: nn.Module,
    ways: int,
    num_episodes: int,
) -> Dict[str, float]:
    """Evaluate on multiple episodes."""
    encoder.eval()
    transformer.eval()
    losses: List[float] = []
    accuracies: List[float] = []
    eta_values: List[float] = []
    plastic_values: List[float] = []
    
    # Confusion matrix
    confusion_matrix = np.zeros((ways, ways), dtype=np.int64)
    
    # Use gradient context if using gradient-based plasticity
    context = torch.enable_grad if transformer.rule == "gradient" else torch.no_grad
    
    with context():
        for episode_idx in range(num_episodes):
            if episode_idx % 10 == 0:
                print(f"  Evaluating episode {episode_idx+1}/{num_episodes}", flush=True)
            
            # Sample episode
            support_strings, support_labels, query_strings, query_labels, category_names = task.sample_episode()
            
            # Encode strings
            support_embeddings = encoder(support_strings)
            query_embeddings = encoder(query_strings)
            
            support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
            query_labels_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)
            
            state = transformer.init_state(device)
            loss = 0.0
            correct = 0
            total = 0
            
            # Process support set
            for idx in range(support_embeddings.shape[0]):
                embedding = support_embeddings[idx]
                label = support_labels_tensor[idx]
                one_hot = F.one_hot(label, num_classes=ways).float()
                step_vec = build_support_vector(embedding, one_hot)
                outputs = transformer.forward_step(step_vec, state)
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
            
            # Process query set
            for idx in range(query_embeddings.shape[0]):
                embedding = query_embeddings[idx]
                label = query_labels_tensor[idx]
                step_vec = build_query_vector(embedding, ways)
                outputs = transformer.forward_step(step_vec, state)
                logits = outputs["logits"]
                loss += loss_fn(logits.unsqueeze(0), label.unsqueeze(0)).item()
                pred = logits.argmax().item()
                true_label = label.item()
                correct += int(pred == true_label)
                total += 1
                
                # Update confusion matrix
                confusion_matrix[true_label, pred] += 1
                
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
            
            episode_loss = loss / max(total, 1)
            episode_accuracy = correct / max(total, 1)
            losses.append(episode_loss)
            accuracies.append(episode_accuracy)
    
    eta_mean, eta_std = summarise(eta_values)
    plastic_mean, plastic_std = summarise(plastic_values)
    
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": sum(accuracies) / max(len(accuracies), 1),
        "eta_mean": eta_mean,
        "eta_std": eta_std,
        "plastic_norm_mean": plastic_mean,
        "plastic_norm_std": plastic_std,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def execute_single_run(args: argparse.Namespace, device: torch.device, seed: int) -> Dict:
    """Execute a single experimental run with given seed."""
    set_seed(seed)
    print(f"[seed={seed}] Starting single run...", flush=True)
    
    # Create tasks
    train_task = WordCategoryTask(
        WordCategoryConfig(
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="train",
        )
    )
    val_task = WordCategoryTask(
        WordCategoryConfig(
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="test",
        )
    )
    
    # Create encoder and transformer
    if args.use_word2vec_encoder:
        print(f"[seed={seed}] Using Word2VecEncoder with GloVe embeddings", flush=True)
        encoder = Word2VecEncoder(
            vocabulary=ALL_WORDS,
            output_dim=args.embedding_dim,
            glove_path=args.glove_path,
            glove_dim=args.glove_dim,
        ).to(device)
    elif args.use_semantic_encoder:
        print(f"[seed={seed}] Using SimpleSemanticEncoder (category-based embeddings)", flush=True)
        encoder = SimpleSemanticEncoder(
            vocabulary=ALL_WORDS,
            categories=CATEGORIES,
            output_dim=args.embedding_dim,
        ).to(device)
    elif args.use_onehot_encoder:
        print(f"[seed={seed}] Using OneHotTextEncoder with vocabulary size {len(ALL_WORDS)}", flush=True)
        encoder = OneHotTextEncoder(
            vocabulary=ALL_WORDS,
            output_dim=args.embedding_dim,
        ).to(device)
    else:
        print(f"[seed={seed}] Using CharacterEncoder", flush=True)
        encoder = CharacterEncoder(
            output_dim=args.embedding_dim,
            char_embed_dim=args.char_embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.encoder_layers,
            dropout=args.dropout,
        ).to(device)
    
    transformer = PlasticTransformerModel(
        input_dim=args.embedding_dim + args.ways + 1,
        output_dim=args.ways,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        aux_dim=args.aux_dim,
        rule=args.rule,
        eta0=args.eta0,
        max_norm=args.max_norm,
    ).to(device)
    
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
    )
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    
    history: List[Dict[str, float]] = []
    for epoch in range(train_cfg.epochs):
        print(f"[seed={seed}] Starting epoch {epoch+1}/{train_cfg.epochs}...", flush=True)
        train_loss, train_diag = train_epoch(
            encoder,
            transformer,
            train_task,
            device,
            optimizer,
            loss_fn,
            train_cfg,
            args.ways,
        )
        print(f"[seed={seed}] Finished training for epoch {epoch+1}/{train_cfg.epochs}.", flush=True)
        metrics = evaluate(
            encoder,
            transformer,
            val_task,
            device,
            loss_fn,
            args.ways,
            train_cfg.val_episodes,
        )
        print(f"[seed={seed}] Finished evaluation for epoch {epoch+1}/{train_cfg.epochs}.", flush=True)
        metrics["train_loss"] = train_loss
        metrics["train_eta_mean"] = train_diag["eta_mean"]
        metrics["train_eta_std"] = train_diag["eta_std"]
        metrics["train_plastic_norm_mean"] = train_diag["plastic_norm_mean"]
        metrics["train_plastic_norm_std"] = train_diag["plastic_norm_std"]
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(
            f"[seed={seed}] Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}",
            flush=True
        )
    
    return {
        "history": history,
        "final": history[-1],
        "task": {
            "train": asdict(train_task.config),
            "val": asdict(val_task.config),
        },
        "training": asdict(train_cfg),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    """Run full experiment with multiple seeds."""
    print('Running word category classification experiment...', flush=True)
    device = resolve_device(args.device)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    run_summaries = []
    final_metrics: List[Dict[str, float]] = []
    task_config = None
    training_config = None
    
    for seed in seeds:
        summary = execute_single_run(args, device, seed)
        run_summaries.append({"seed": seed, "history": summary["history"], "final": summary["final"]})
        final_metrics.append(summary["final"])
        if task_config is None:
            task_config = summary["task"]
        if training_config is None:
            training_config = summary["training"]
    
    aggregate = aggregate_metrics(final_metrics)
    last_final = final_metrics[-1] if final_metrics else {}
    
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
                    "embedding_dim": args.embedding_dim,
                    "encoder": {
                        "char_embed_dim": args.char_embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "encoder_layers": args.encoder_layers,
                    },
                },
                "task": task_config,
                "training": training_config,
            },
            "runs": run_summaries,
            "aggregate": aggregate,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    
    return last_final


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Word category classification experiment")
    
    # Model arguments
    parser.add_argument("--rule", choices=["none", "hebbian", "gradient"], default="gradient",
                        help="Plasticity rule")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Text embedding dimension")
    parser.add_argument("--char-embed-dim", type=int, default=32,
                        help="Character embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="LSTM hidden dimension")
    parser.add_argument("--encoder-layers", type=int, default=2,
                        help="Number of LSTM layers in encoder")
    parser.add_argument("--model-dim", type=int, default=256,
                        help="Transformer model dimension")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--ffn-dim", type=int, default=512,
                        help="FFN intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--aux-dim", type=int, default=4,
                        help="Auxiliary dimension for gradient rule")
    parser.add_argument("--eta0", type=float, default=0.2,
                        help="Initial plasticity learning rate")
    parser.add_argument("--max-norm", type=float, default=1.0,
                        help="Maximum norm for plastic weights")
    
    # Task arguments
    parser.add_argument("--ways", type=int, default=3,
                        help="Number of categories per episode")
    parser.add_argument("--shots", type=int, default=1,
                        help="Number of support examples per category")
    parser.add_argument("--queries", type=int, default=2,
                        help="Number of query examples per category")
    parser.add_argument("--use-word2vec-encoder", action="store_true",
                        help="Use Word2VecEncoder with pre-trained GloVe embeddings")
    parser.add_argument("--glove-path", type=str, default="embeddings/glove.6B.100d.txt",
                        help="Path to GloVe embeddings file")
    parser.add_argument("--glove-dim", type=int, default=100,
                        help="Dimension of GloVe embeddings (50, 100, 200, or 300)")
    parser.add_argument("--use-semantic-encoder", action="store_true",
                        help="Use SimpleSemanticEncoder (category-based embeddings with semantic structure)")
    parser.add_argument("--use-onehot-encoder", action="store_true",
                        help="Use OneHotTextEncoder (independent random embeddings)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, default=80,
                        help="Number of episodes per epoch")
    parser.add_argument("--val-episodes", type=int, default=50,
                        help="Number of validation episodes")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--clip-norm", type=float, default=5.0,
                        help="Gradient clipping norm")
    
    # Experiment arguments
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Device to use")
    parser.add_argument("--base-seed", type=int, default=123,
                        help="Base random seed")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds to run")
    parser.add_argument("--output-path", type=str, default="",
                        help="Path to save results JSON")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f'Starting word category classification experiment with args: {args}', flush=True)
    run_experiment(args)
