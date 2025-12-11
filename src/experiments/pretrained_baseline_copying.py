"""
Evaluate pre-trained transformer baseline on copying task.

This script tests a frozen BERT model on the copying task to compare
against the plastic transformer's in-context learning.
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from src.tasks.copying import CopyingTaskConfig, CopyingTaskDataset, generate_copying_batch


class PretrainedTransformerForCopying(nn.Module):
    """
    Pre-trained transformer baseline for copying task.
    Uses frozen BERT with learnable token embeddings and output head.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        vocab_size: int = 8,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Load pre-trained transformer
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Learnable embeddings for copying task tokens
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        
        # Project token embeddings to BERT input dimension
        bert_dim = self.backbone.config.hidden_size
        self.input_projection = nn.Linear(hidden_dim, bert_dim)
        
        # Output head to predict next tokens
        self.output_head = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_seq: (batch_size, seq_length) tensor of token indices
            
        Returns:
            logits: (batch_size, seq_length, vocab_size) prediction logits
        """
        # Embed tokens
        token_embeds = self.token_embeddings(input_seq)  # (B, L, hidden_dim)
        
        # Project to BERT dimension
        projected = self.input_projection(token_embeds)  # (B, L, bert_dim)
        
        # Pass through BERT
        outputs = self.backbone(inputs_embeds=projected)
        hidden_states = outputs.last_hidden_state  # (B, L, bert_dim)
        
        # Predict output tokens
        logits = self.output_head(hidden_states)  # (B, L, vocab_size)
        
        return logits


def evaluate_copying(
    model: PretrainedTransformerForCopying,
    dataset: CopyingTaskDataset,
    num_batches: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on copying task.
    
    Args:
        model: Pre-trained baseline model
        dataset: Copying task dataset
        num_batches: Number of batches to evaluate
        batch_size: Batch size
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{num_batches}", flush=True)
            
            # Generate batch
            indices = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            inputs, targets = generate_copying_batch(dataset, indices, device)
            
            # Forward pass
            logits = model(inputs)  # (B, L, vocab_size)
            
            # Reshape for loss computation
            B, L, V = logits.shape
            logits_flat = logits.view(B * L, V)
            targets_flat = targets.view(B * L)
            
            # Compute loss
            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()
            
            # Compute accuracy (only on recall phase)
            # Recall phase is the last seq_length tokens
            seq_length = dataset.config.seq_length
            recall_start = L - seq_length
            
            recall_logits = logits[:, recall_start:, :]  # (B, seq_length, vocab_size)
            recall_targets = targets[:, recall_start:]  # (B, seq_length)
            
            predictions = recall_logits.argmax(dim=-1)  # (B, seq_length)
            correct = (predictions == recall_targets).sum().item()
            
            total_correct += correct
            total_tokens += B * seq_length
    
    avg_loss = total_loss / (num_batches * batch_size * dataset.total_length)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "total_tokens": total_tokens,
    }


def run_baseline_experiment(args: argparse.Namespace) -> None:
    """Run baseline experiment on copying task."""
    print(f"Starting pre-trained baseline experiment on copying task")
    print(f"Model: {args.model_name}")
    print(f"Configuration: seq_length={args.seq_length}, delay={args.delay}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print(f"\nLoading pre-trained model: {args.model_name}")
    model = PretrainedTransformerForCopying(
        model_name=args.model_name,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        freeze_backbone=not args.finetune,
    ).to(device)
    
    if args.finetune:
        print("Fine-tuning mode: backbone weights trainable")
    else:
        print("Frozen mode: backbone weights fixed")
    
    # Create dataset
    print("\nSetting up copying task...")
    config = CopyingTaskConfig(
        seq_length=args.seq_length,
        delay=args.delay,
        vocab_size=args.vocab_size,
        dataset_size=args.num_batches * args.batch_size,
        device=str(device),
    )
    dataset = CopyingTaskDataset(config)
    
    # Evaluate
    print(f"\nEvaluating on {args.num_batches} batches (batch_size={args.batch_size})...")
    metrics = evaluate_copying(model, dataset, args.num_batches, args.batch_size, device)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy (recall phase): {metrics['accuracy']:.4f}")
    print(f"Total tokens evaluated: {metrics['total_tokens']}")
    print("="*70)
    
    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results = {
            "config": {
                "model_name": args.model_name,
                "hidden_dim": args.hidden_dim,
                "finetune": args.finetune,
                "task": {
                    "seq_length": args.seq_length,
                    "delay": args.delay,
                    "vocab_size": args.vocab_size,
                },
                "evaluation": {
                    "num_batches": args.num_batches,
                    "batch_size": args.batch_size,
                },
            },
            "metrics": metrics,
        }
        
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-trained Transformer Baseline - Copying Task")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for token embeddings")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune backbone (default: freeze)")
    
    # Task arguments
    parser.add_argument("--seq-length", type=int, default=5,
                        help="Length of sequence to copy")
    parser.add_argument("--delay", type=int, default=20,
                        help="Number of delay steps before recall")
    parser.add_argument("--vocab-size", type=int, default=8,
                        help="Vocabulary size")
    
    # Evaluation arguments
    parser.add_argument("--num-batches", type=int, default=100,
                        help="Number of batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use")
    
    # Output
    parser.add_argument("--output-path", type=str,
                        default="experiments/results/copying_baseline_bert.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    run_baseline_experiment(args)


if __name__ == "__main__":
    main()
