"""
Plot results for word category classification experiments.

Generates:
1. Bar plot comparing final accuracy across plasticity rules
2. Line plot showing accuracy vs. epoch for each rule
3. Mechanistic traces (eta and plastic_norm over training)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("experiments/word_category_results")
FIGURES_DIR = Path("word_category_figures")


@dataclass
class ResultRecord:
    file: Path
    rule: str
    shots: int
    aggregate: Dict[str, float]
    runs: List[Dict]


def load_word_category_results() -> List[ResultRecord]:
    """Load all word category classification result files."""
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Results directory '{RESULTS_DIR}' not found.")
    
    records: List[ResultRecord] = []
    
    # Look for word_category result files
    for path in sorted(RESULTS_DIR.glob("word_category_*shot_*_semantic*.json")):
        if "glove" in path.name.lower():
            continue  # Skip GloVe results for now
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        config = data.get("config", {}) or {}
        aggregate = data.get("aggregate", {}) or {}
        runs = data.get("runs", []) or []
        
        # Extract rule
        model_cfg = config.get("model", {}) or {}
        rule = model_cfg.get("rule", "unknown")
        
        # Extract shots
        task_cfg = config.get("task", {}) or {}
        train_cfg = task_cfg.get("train", {}) or {}
        shots = train_cfg.get("shots", 1)
        
        records.append(
            ResultRecord(
                file=path,
                rule=rule,
                shots=shots,
                aggregate=aggregate,
                runs=runs,
            )
        )
    
    return records


def group_by_shots_and_rule(records: List[ResultRecord]) -> Dict[int, Dict[str, ResultRecord]]:
    """Group records by number of shots and rule."""
    grouped: Dict[int, Dict[str, ResultRecord]] = {}
    for record in records:
        grouped.setdefault(record.shots, {})
        grouped[record.shots][record.rule] = record
    return grouped


def plot_accuracy_bars(grouped: Dict[int, Dict[str, ResultRecord]]) -> None:
    """Plot bar chart comparing final accuracy across rules for each shot configuration."""
    rules = ["gradient", "hebbian", "none"]
    rule_labels = {"gradient": "Gradient", "hebbian": "Hebbian", "none": "None"}
    colors = {"gradient": "#1f77b4", "hebbian": "#ff7f0e", "none": "#2ca02c"}
    
    # Get all shot configurations
    shot_configs = sorted(grouped.keys())
    
    fig, axes = plt.subplots(1, len(shot_configs), figsize=(4 * len(shot_configs), 4), sharey=True)
    if len(shot_configs) == 1:
        axes = [axes]
    
    for ax, shots in zip(axes, shot_configs):
        records = grouped.get(shots, {})
        
        values = []
        errors = []
        labels = []
        bar_colors = []
        
        for rule in rules:
            record = records.get(rule)
            if not record:
                continue
            labels.append(rule_labels[rule])
            values.append(record.aggregate.get("accuracy_mean", np.nan))
            errors.append(record.aggregate.get("accuracy_std", 0.0))
            bar_colors.append(colors.get(rule, "#555555"))
        
        positions = np.arange(len(values))
        ax.bar(positions, values, yerr=errors, color=bar_colors, capsize=5, width=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(f"Word Category ({shots}-shot)")
        ax.set_ylabel("Accuracy" if shots == shot_configs[0] else "")
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if shots == shot_configs[0]:
            ax.legend(frameon=False, loc='upper left', fontsize=8)
    
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "word_category_accuracy_bars.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "word_category_accuracy_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'word_category_accuracy_bars.png'}")


def aggregate_epoch_series(record: ResultRecord, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metric values across runs by epoch."""
    epoch_values: Dict[int, List[float]] = {}
    
    for run in record.runs:
        for entry in run.get("history", []):
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            value = entry.get(metric)
            if value is None:
                continue
            epoch_values.setdefault(epoch, []).append(value)
    
    if not epoch_values:
        return np.array([]), np.array([]), np.array([])
    
    epochs = sorted(epoch_values.keys())
    means = np.array([np.mean(epoch_values[e]) for e in epochs])
    stds = np.array([np.std(epoch_values[e]) for e in epochs])
    
    return np.array(epochs), means, stds


def plot_accuracy_curves(grouped: Dict[int, Dict[str, ResultRecord]]) -> None:
    """Plot accuracy vs. epoch for each rule and shot configuration."""
    rules = ["gradient", "hebbian", "none"]
    rule_labels = {"gradient": "Gradient", "hebbian": "Hebbian", "none": "None"}
    colors = {"gradient": "#1f77b4", "hebbian": "#ff7f0e", "none": "#2ca02c"}
    
    shot_configs = sorted(grouped.keys())
    
    fig, axes = plt.subplots(1, len(shot_configs), figsize=(5 * len(shot_configs), 4), sharey=True)
    if len(shot_configs) == 1:
        axes = [axes]
    
    for ax, shots in zip(axes, shot_configs):
        records = grouped.get(shots, {})
        
        for rule in rules:
            record = records.get(rule)
            if not record:
                continue
            
            epochs, means, stds = aggregate_epoch_series(record, "accuracy")
            if epochs.size == 0:
                continue
            
            ax.plot(epochs, means, label=rule_labels[rule], color=colors[rule], marker='o', linewidth=2)
            ax.fill_between(epochs, means - stds, means + stds, color=colors[rule], alpha=0.2)
        
        ax.set_title(f"Word Category ({shots}-shot)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy" if shots == shot_configs[0] else "")
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False, loc='lower right')
    
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "word_category_accuracy_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "word_category_accuracy_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'word_category_accuracy_curves.png'}")


def plot_mechanistic_traces(grouped: Dict[int, Dict[str, ResultRecord]]) -> None:
    """Plot mechanistic traces (eta and plastic_norm) over training."""
    rules = ["gradient", "hebbian", "none"]
    rule_labels = {"gradient": "Gradient", "hebbian": "Hebbian", "none": "None"}
    colors = {"gradient": "#1f77b4", "hebbian": "#ff7f0e", "none": "#2ca02c"}
    
    metrics = [
        ("eta_mean", "Neuromodulation $\\eta(t)$"),
        ("plastic_norm_mean", "Plastic Weight Norm"),
    ]
    
    shot_configs = sorted(grouped.keys())
    
    num_rows = len(metrics)
    num_cols = len(shot_configs)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3.5 * num_rows), sharex=False)
    
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for col, shots in enumerate(shot_configs):
        records = grouped.get(shots, {})
        
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            
            for rule in rules:
                record = records.get(rule)
                if not record:
                    continue
                
                epochs, means, stds = aggregate_epoch_series(record, metric)
                if epochs.size == 0:
                    continue
                
                ax.plot(epochs, means, label=rule_labels[rule], color=colors[rule], marker='o', linewidth=2)
                ax.fill_between(epochs, means - stds, means + stds, color=colors[rule], alpha=0.2)
            
            if row == 0:
                ax.set_title(f"Word Category ({shots}-shot)")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.3)
            
            # Add legend to top-right plot
            if row == 0 and col == num_cols - 1:
                ax.legend(frameon=False, loc='upper right')
    
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "word_category_mechanistic_traces.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "word_category_mechanistic_traces.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'word_category_mechanistic_traces.png'}")


def print_summary(grouped: Dict[int, Dict[str, ResultRecord]]) -> None:
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Word Category Classification Results Summary")
    print("="*60)
    
    for shots in sorted(grouped.keys()):
        print(f"\n{shots}-shot configuration:")
        records = grouped[shots]
        
        for rule in ["gradient", "hebbian", "none"]:
            record = records.get(rule)
            if not record:
                continue
            
            acc_mean = record.aggregate.get("accuracy_mean", np.nan)
            acc_std = record.aggregate.get("accuracy_std", 0.0)
            loss_mean = record.aggregate.get("loss_mean", np.nan)
            
            print(f"  {rule:10s}: accuracy={acc_mean:.3f}±{acc_std:.3f}, loss={loss_mean:.3f}")
    
    print("\n" + "="*60 + "\n")


def main() -> None:
    print("Loading word category classification results...")
    records = load_word_category_results()
    
    if not records:
        print("No word category results found!")
        print("Looking for files matching: word_category_*shot_*_semantic*.json")
        return
    
    print(f"Found {len(records)} result files")
    
    grouped = group_by_shots_and_rule(records)
    
    print_summary(grouped)
    
    FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    plot_accuracy_bars(grouped)
    plot_accuracy_curves(grouped)
    plot_mechanistic_traces(grouped)
    
    print(f"\nAll figures saved to '{FIGURES_DIR}/'")


if __name__ == "__main__":
    main()
