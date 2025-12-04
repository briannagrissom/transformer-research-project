"""
Utility script for plotting denoising text experiment results.
Generates confusion matrices, accuracy curves, and noise analysis plots.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(confusion_matrix: np.ndarray, output_path: str, title: str = "Confusion Matrix"):
    """
    Plot and save a confusion matrix heatmap.
    
    Args:
        confusion_matrix: 2D array of shape (ways, ways)
        output_path: Path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize by row (true labels)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums != 0
    )
    
    sns.heatmap(
        normalized_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Normalized Count"},
        xticklabels=range(confusion_matrix.shape[1]),
        yticklabels=range(confusion_matrix.shape[0]),
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_accuracy_curve(history: List[Dict], output_path: str, title: str = "Accuracy vs Epoch"):
    """
    Plot accuracy over training epochs.
    
    Args:
        history: List of metric dictionaries from training
        output_path: Path to save the figure
        title: Title for the plot
    """
    epochs = [h["epoch"] for h in history]
    accuracies = [h["accuracy"] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve to {output_path}")


def plot_loss_curve(history: List[Dict], output_path: str, title: str = "Loss vs Epoch"):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history: List of metric dictionaries from training
        output_path: Path to save the figure
        title: Title for the plot
    """
    epochs = [h["epoch"] for h in history]
    train_losses = [h.get("train_loss", 0) for h in history]
    val_losses = [h.get("loss", 0) for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss", linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, marker="s", label="Val Loss", linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_plasticity_diagnostics(history: List[Dict], output_path: str, title: str = "Plasticity Diagnostics"):
    """
    Plot eta and plastic norm over epochs.
    
    Args:
        history: List of metric dictionaries from training
        output_path: Path to save the figure
        title: Title for the plot
    """
    epochs = [h["epoch"] for h in history]
    eta_means = [h.get("eta_mean", 0) for h in history]
    plastic_norms = [h.get("plastic_norm_mean", 0) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot eta
    ax1.plot(epochs, eta_means, marker="o", linewidth=2, markersize=6, color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("η (Plasticity Learning Rate)")
    ax1.set_title("Plasticity Learning Rate")
    ax1.grid(True, alpha=0.3)
    
    # Plot plastic norm
    ax2.plot(epochs, plastic_norms, marker="s", linewidth=2, markersize=6, color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("||δ||₂ (Plastic Weight Norm)")
    ax2.set_title("Plastic Weight Norm")
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plasticity diagnostics to {output_path}")


def plot_edit_distance(history: List[Dict], output_path: str, title: str = "Edit Distance vs Epoch"):
    """
    Plot edit distance (noise level) over epochs.
    
    Args:
        history: List of metric dictionaries from training
        output_path: Path to save the figure
        title: Title for the plot
    """
    epochs = [h["epoch"] for h in history]
    edit_distances = [h.get("edit_distance_mean", 0) for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, edit_distances, marker="o", linewidth=2, markersize=6, color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Character Error Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved edit distance curve to {output_path}")


def plot_noise_level_comparison(results_files: List[str], output_path: str):
    """
    Compare accuracy across different noise levels.
    
    Args:
        results_files: List of paths to result JSON files (one per noise level)
        output_path: Path to save the figure
    """
    noise_levels = []
    accuracies = []
    accuracies_std = []
    
    for file_path in results_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract noise probability from config
        noise_prob = data["config"]["task"]["train"]["noise_prob"]
        noise_levels.append(noise_prob)
        
        # Extract final accuracy (mean and std across seeds)
        acc_mean = data["aggregate"].get("accuracy_mean", 0)
        acc_std = data["aggregate"].get("accuracy_std", 0)
        accuracies.append(acc_mean)
        accuracies_std.append(acc_std)
    
    # Sort by noise level
    sorted_indices = np.argsort(noise_levels)
    noise_levels = [noise_levels[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    accuracies_std = [accuracies_std[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        noise_levels,
        accuracies,
        yerr=accuracies_std,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
    )
    plt.xlabel("Noise Probability (p)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise Level")
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved noise level comparison to {output_path}")


def visualize_results(result_path: str, output_dir: str):
    """
    Generate all visualizations for a single experiment result.
    
    Args:
        result_path: Path to the result JSON file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(result_path, "r") as f:
        data = json.load(f)
    
    # Extract experiment info
    rule = data["config"]["model"]["rule"]
    noise_type = data["config"]["task"]["train"]["noise_type"]
    noise_prob = data["config"]["task"]["train"]["noise_prob"]
    
    base_name = f"{rule}_{noise_type}_p{noise_prob}"
    
    # Use the last seed's history for time-series plots
    last_run = data["runs"][-1]
    history = last_run["history"]
    
    # Use the final confusion matrix from the last seed
    final_metrics = last_run["final"]
    confusion_matrix = np.array(final_metrics.get("confusion_matrix", []))
    
    # Generate plots
    plot_accuracy_curve(
        history,
        str(output_dir / f"{base_name}_accuracy.png"),
        title=f"Accuracy vs Epoch ({rule}, {noise_type}, p={noise_prob})"
    )
    
    plot_loss_curve(
        history,
        str(output_dir / f"{base_name}_loss.png"),
        title=f"Loss vs Epoch ({rule}, {noise_type}, p={noise_prob})"
    )
    
    plot_plasticity_diagnostics(
        history,
        str(output_dir / f"{base_name}_plasticity.png"),
        title=f"Plasticity Diagnostics ({rule}, {noise_type}, p={noise_prob})"
    )
    
    plot_edit_distance(
        history,
        str(output_dir / f"{base_name}_edit_distance.png"),
        title=f"Edit Distance vs Epoch ({rule}, {noise_type}, p={noise_prob})"
    )
    
    if confusion_matrix.size > 0:
        plot_confusion_matrix(
            confusion_matrix,
            str(output_dir / f"{base_name}_confusion_matrix.png"),
            title=f"Confusion Matrix ({rule}, {noise_type}, p={noise_prob})"
        )
    
    print(f"\nAll plots saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize denoising text experiment results")
    parser.add_argument("--result-path", type=str, required=True,
                        help="Path to result JSON file")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Directory to save plots")
    parser.add_argument("--compare-noise", nargs="+", type=str,
                        help="Paths to multiple result files for noise level comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Generate visualizations for the main result
    visualize_results(args.result_path, args.output_dir)
    
    # If comparing noise levels, generate comparison plot
    if args.compare_noise:
        output_dir = Path(args.output_dir)
        plot_noise_level_comparison(
            args.compare_noise,
            str(output_dir / "noise_level_comparison.png")
        )
