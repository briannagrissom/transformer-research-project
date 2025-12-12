"""
Evaluate pre-trained transformer baseline on one-shot classification.

This script tests a frozen vision transformer (ViT or CLIP) on image classification
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

from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor

from src.tasks.few_shot_classification import FewShotClassificationConfig, FewShotClassificationTask


class PretrainedVisionTransformerBaseline(nn.Module):
    """
    Pre-trained vision transformer baseline for few-shot classification.
    Uses frozen CLIP or ViT with prototype-based classification.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        # Determine if using CLIP or regular vision model
        self.is_clip = "clip" in model_name.lower()
        
        # Load pre-trained model and processor
        # Use safetensors format to avoid PyTorch security vulnerability
        if self.is_clip:
            self.backbone = CLIPModel.from_pretrained(
                model_name,
                use_safetensors=True
            ).vision_model
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True
            )
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        if self.is_clip:
            backbone_dim = self.backbone.config.hidden_size
        else:
            backbone_dim = self.backbone.config.hidden_size
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def encode_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: (N, C, H, W) tensor of images
            device: Device to run on
            
        Returns:
            embeddings: (N, output_dim) tensor of embeddings
        """
        # Convert grayscale to RGB if needed (for models expecting 3 channels)
        if images.shape[1] == 1:  # Grayscale
            images = images.repeat(1, 3, 1, 1)  # Repeat channel dimension
        
        # Resize images to match model's expected size (224x224 for CLIP)
        if self.is_clip:
            expected_size = 224
        else:
            # For other vision transformers, check config
            if hasattr(self.backbone.config, 'image_size'):
                expected_size = self.backbone.config.image_size
            else:
                expected_size = 224  # default
        
        # Resize if needed
        _, _, h, w = images.shape
        if h != expected_size or w != expected_size:
            images = F.interpolate(
                images, 
                size=(expected_size, expected_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Forward through backbone
        if self.is_clip:
            outputs = self.backbone(pixel_values=images)
            # Use pooled output
            features = outputs.pooler_output  # (N, hidden_dim)
        else:
            outputs = self.backbone(pixel_values=images)
            # Use [CLS] token or pooled output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project to output dimension
        embeddings = self.projection(features)
        
        return embeddings
    
    def forward_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        ways: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Process one episode using prototype-based classification.
        
        Args:
            support_images: (ways * shots, C, H, W) support images
            support_labels: (ways * shots,) support labels
            query_images: (ways * queries, C, H, W) query images
            ways: Number of classes
            device: Device to run on
            
        Returns:
            logits: (ways * queries, ways) classification logits
        """
        # Encode all images
        support_embeddings = self.encode_images(support_images, device)
        query_embeddings = self.encode_images(query_images, device)
        
        # Compute class prototypes (mean of support embeddings per class)
        prototypes = []
        for label in range(ways):
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (ways, output_dim)
        
        # Compute distances from queries to prototypes
        # Negative distance as logits (closer = higher logit)
        distances = torch.cdist(query_embeddings, prototypes)  # (num_queries, ways)
        logits = -distances
        
        return logits


class PretrainedBaselineForClassification(nn.Module):
    """Wrapper for classification task."""
    
    def __init__(self, model_name: str, output_dim: int, freeze_backbone: bool):
        super().__init__()
        self.model = PretrainedVisionTransformerBaseline(
            model_name=model_name,
            output_dim=output_dim,
            freeze_backbone=freeze_backbone,
        )
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        ways: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.model.forward_episode(
            support_images, support_labels, query_images, ways, device
        )


def evaluate_baseline(
    model: PretrainedBaselineForClassification,
    task: FewShotClassificationTask,
    num_episodes: int,
    device: torch.device,
    ways: int,
) -> Dict[str, float]:
    """
    Evaluate baseline model on classification task.
    
    Args:
        model: Pre-trained baseline model
        task: Classification task
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
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for episode_idx in range(num_episodes):
            if episode_idx % 10 == 0:
                print(f"  Episode {episode_idx+1}/{num_episodes}", flush=True)
            
            # Sample episode
            support_images, support_labels, query_images, query_labels, _ = task.sample_episode()
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels_tensor = query_labels.to(device)
            
            # Forward pass
            logits = model(
                support_images=support_images,
                support_labels=support_labels,
                query_images=query_images,
                ways=ways,
                device=device,
            )
            
            # Compute loss and accuracy
            loss = loss_fn(logits, query_labels_tensor).item()
            
            predictions = logits.argmax(dim=1).cpu().numpy()
            query_labels_np = query_labels_tensor.cpu().numpy()
            correct = (predictions == query_labels_np).sum()
            accuracy = correct / len(query_labels_np)
            
            losses.append(loss)
            accuracies.append(accuracy)
            
            # Update confusion matrix
            for true_label, pred_label in zip(query_labels_np, predictions):
                confusion_matrix[true_label, pred_label] += 1
    
    return {
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "confusion_matrix": confusion_matrix.tolist(),
    }


def run_baseline_experiment(args: argparse.Namespace) -> None:
    """Run baseline experiment."""
    print(f"Starting pre-trained baseline experiment on {args.dataset} classification")
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
    model = PretrainedBaselineForClassification(
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
    train_task = FewShotClassificationTask(
        FewShotClassificationConfig(
            root=args.data_root,
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="train",
            dataset=args.dataset,
        )
    )
    test_task = FewShotClassificationTask(
        FewShotClassificationConfig(
            root=args.data_root,
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="test",
            dataset=args.dataset,
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
    
    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results = {
            "config": {
                "model_name": args.model_name,
                "output_dim": args.output_dim,
                "finetune": args.finetune,
                "task": {
                    "dataset": args.dataset,
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
    parser = argparse.ArgumentParser(description="Pre-trained Transformer Baseline - Classification")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace model name (CLIP or ViT)")
    parser.add_argument("--output-dim", type=int, default=256,
                        help="Output dimension for projection head")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune backbone (default: freeze)")
    
    # Task arguments
    parser.add_argument("--dataset", type=str, default="omniglot",
                        choices=["omniglot", "cifarfs"],
                        help="Dataset to use")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--ways", type=int, default=5,
                        help="Number of classes per episode")
    parser.add_argument("--shots", type=int, default=1,
                        help="Number of support examples per class")
    parser.add_argument("--queries", type=int, default=15,
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
                        default="experiments/results/classification_baseline_clip.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    run_baseline_experiment(args)


if __name__ == "__main__":
    main()
