"""
Evaluate pre-trained transformer baseline on word category classification.

This script tests a frozen BERT model using prototype-based classification
to compare against the plastic transformer's in-context learning.
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pretrained_baseline import PretrainedBaselineForWordCategory
from src.tasks.word_category_classification import WordCategoryConfig, WordCategoryTask


def evaluate_baseline(
    model: PretrainedBaselineForWordCategory,
    task: WordCategoryTask,
    num_episodes: int,
    device: torch.device,
    ways: int,
) -> Dict[str, float]:
    """
    Evaluate baseline model on word category task.
    
    Args:
        model: Pre-trained baseline model
        task: Word category task
        num_episodes: Number of episodes to evaluate
        device: Device to run on
        ways: Number of classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    losses = []
    accuracies = []
    confusion_matrix = np.zeros((ways, ways), dtype=np.int64)
    
    # Category-specific confusion matrix
    # Map semantic categories to indices: animals=0, clothing=1, furniture=2
    category_to_idx = {"animals": 0, "clothing": 1, "furniture": 2}
    semantic_confusion = np.zeros((3, 3), dtype=np.int64)
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for episode_idx in range(num_episodes):
            if episode_idx % 10 == 0:
                print(f"  Episode {episode_idx+1}/{num_episodes}", flush=True)
            
            # Sample episode - now capture category_map
            support_strings, support_labels, query_strings, query_labels, category_map = task.sample_episode()
            
            # Forward pass
            logits = model(
                support_strings=support_strings,
                support_labels=support_labels,
                query_strings=query_strings,
                ways=ways,
                device=device,
            )
            
            # Compute loss and accuracy
            query_labels_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)
            loss = loss_fn(logits, query_labels_tensor).item()
            
            predictions = logits.argmax(dim=1).cpu().numpy()
            correct = (predictions == np.array(query_labels)).sum()
            accuracy = correct / len(query_labels)
            
            losses.append(loss)
            accuracies.append(accuracy)
            
            # Update episode-local confusion matrix
            for true_label, pred_label in zip(query_labels, predictions):
                confusion_matrix[true_label, pred_label] += 1
            
            # Update semantic confusion matrix
            # category_map maps episode label -> category name
            for true_label, pred_label in zip(query_labels, predictions):
                true_category = category_map[true_label]
                pred_category = category_map[pred_label]
                true_idx = category_to_idx[true_category]
                pred_idx = category_to_idx[pred_category]
                semantic_confusion[true_idx, pred_idx] += 1
    
    return {
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "confusion_matrix": confusion_matrix.tolist(),
        "semantic_confusion_matrix": semantic_confusion.tolist(),
        "semantic_categories": ["animals", "clothing", "furniture"],
    }


def run_baseline_experiment(args: argparse.Namespace) -> None:
    """Run baseline experiment."""
    print(f"Starting pre-trained baseline experiment on word category classification")
    print(f"Model: {args.model_name}")
    print(f"Configuration: {args.ways}-way, {args.shots}-shot")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print(f"\nLoading pre-trained model: {args.model_name}")
    model = PretrainedBaselineForWordCategory(
        model_name=args.model_name,
        output_dim=args.output_dim,
        freeze_backbone=not args.finetune,
    ).to(device)
    
    if args.finetune:
        print("Fine-tuning mode: backbone weights trainable")
    else:
        print("Frozen mode: backbone weights fixed")
    
    # Create tasks
    print("\nSetting up tasks...")
    train_task = WordCategoryTask(
        WordCategoryConfig(
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="train",
        )
    )
    test_task = WordCategoryTask(
        WordCategoryConfig(
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="test",
        )
    )
    
    # Evaluate on train and test
    print(f"\nEvaluating on {args.train_episodes} training episodes...")
    train_metrics = evaluate_baseline(model, train_task, args.train_episodes, device, args.ways)
    
    print(f"\nEvaluating on {args.test_episodes} test episodes...")
    test_metrics = evaluate_baseline(model, test_task, args.test_episodes, device, args.ways)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test  - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
    print("="*70)
    
    # Print semantic confusion matrix
    print("\nSemantic Confusion Matrix (Test):")
    print("Rows = True category, Columns = Predicted category")
    print("Categories: animals (0), clothing (1), furniture (2)")
    print("-" * 50)
    semantic_conf = np.array(test_metrics['semantic_confusion_matrix'])
    for i, true_cat in enumerate(["animals", "clothing", "furniture"]):
        print(f"{true_cat:10s} | {semantic_conf[i, 0]:4d} {semantic_conf[i, 1]:4d} {semantic_conf[i, 2]:4d}")
    print("-" * 50)
    
    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results = {
            "config": {
                "model_name": args.model_name,
                "output_dim": args.output_dim,
                "finetune": args.finetune,
                "task": {
                    "ways": args.ways,
                    "shots": args.shots,
                    "queries": args.queries,
                },
                "evaluation": {
                    "train_episodes": args.train_episodes,
                    "test_episodes": args.test_episodes,
                },
            },
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-trained Transformer Baseline")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--output-dim", type=int, default=256,
                        help="Output dimension for projection head")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune backbone (default: freeze)")
    
    # Task arguments
    parser.add_argument("--ways", type=int, default=3,
                        help="Number of classes per episode")
    parser.add_argument("--shots", type=int, default=3,
                        help="Number of support examples per class")
    parser.add_argument("--queries", type=int, default=2,
                        help="Number of query examples per class")
    
    # Evaluation arguments
    parser.add_argument("--train-episodes", type=int, default=100,
                        help="Number of training episodes to evaluate")
    parser.add_argument("--test-episodes", type=int, default=100,
                        help="Number of test episodes to evaluate")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use")
    
    # Output
    parser.add_argument("--output-path", type=str,
                        default="experiments/results/word_category_baseline_bert.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    run_baseline_experiment(args)


if __name__ == "__main__":
    main()
