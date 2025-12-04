#!/usr/bin/env python3
"""
Analyze episode details from denoising text experiment results.
Shows support strings, predictions, and accuracies for each episode.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List


def print_episode_summary(episode: Dict, show_queries: bool = False):
    """Print a formatted summary of a single episode."""
    print(f"\n{'='*80}")
    print(f"Episode {episode['episode_idx']}")
    print(f"{'='*80}")
    print(f"Accuracy: {episode['accuracy']:.2%}")
    print(f"Loss: {episode['loss']:.4f}")
    print()
    
    # Show class words and support strings
    print("Classes and Support Strings (noisy):")
    class_words = episode['class_words']
    support_strings = episode['support_strings']
    support_labels = episode['support_labels']
    
    for class_idx, word in enumerate(class_words):
        print(f"  Class {class_idx}: '{word}'")
        # Find support string for this class
        for support_str, support_label in zip(support_strings, support_labels):
            if support_label == class_idx:
                if support_str == word:
                    print(f"    Support: '{support_str}' (no noise)")
                else:
                    print(f"    Support: '{support_str}' (noisy)")
    print()
    
    # Show prediction accuracy breakdown
    query_true = episode['query_true_labels']
    query_pred = episode['query_predictions']
    
    # Count correct/incorrect per class
    class_correct = {i: 0 for i in range(len(class_words))}
    class_total = {i: 0 for i in range(len(class_words))}
    
    for true_label, pred_label in zip(query_true, query_pred):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1
    
    print("Per-Class Query Accuracy:")
    for class_idx, word in enumerate(class_words):
        correct = class_correct[class_idx]
        total = class_total[class_idx]
        acc = correct / total if total > 0 else 0
        print(f"  '{word}' (class {class_idx}): {correct}/{total} = {acc:.2%}")
    
    # Show sample queries if requested
    if show_queries and 'query_strings' in episode:
        print()
        print("Sample Query Strings (first 10):")
        for i, query_str in enumerate(episode['query_strings'][:10]):
            true_label = query_true[i] if i < len(query_true) else "?"
            pred_label = query_pred[i] if i < len(query_pred) else "?"
            status = "✓" if true_label == pred_label else "✗"
            true_word = class_words[true_label] if isinstance(true_label, int) else "?"
            pred_word = class_words[pred_label] if isinstance(pred_label, int) else "?"
            print(f"    {status} '{query_str}' → predicted: '{pred_word}', true: '{true_word}'")


def analyze_results(result_path: str, 
                    num_episodes: int = 5, 
                    show_best: bool = False,
                    show_worst: bool = False,
                    show_queries: bool = False):
    """Analyze episode details from experiment results."""
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    # Get the last run's episode details
    if 'runs' not in data or len(data['runs']) == 0:
        print("No runs found in results file")
        return
    
    last_run = data['runs'][-1]
    if 'final' not in last_run or 'episode_details' not in last_run['final']:
        print("No episode details found in results")
        print("This results file may have been generated before episode tracking was added")
        return
    
    episodes = last_run['final']['episode_details']
    
    print(f"\n{'='*80}")
    print(f"Denoising Text Experiment Episode Analysis")
    print(f"{'='*80}")
    print(f"Results file: {result_path}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Rule: {data['config']['model']['rule']}")
    print(f"Noise type: {data['config']['task']['train']['noise_type']}")
    print(f"Noise prob: {data['config']['task']['train']['noise_prob']}")
    print()
    
    # Sort episodes by accuracy
    sorted_episodes = sorted(episodes, key=lambda e: e['accuracy'])
    
    if show_worst:
        print(f"\n{'#'*80}")
        print(f"WORST {num_episodes} EPISODES (lowest accuracy)")
        print(f"{'#'*80}")
        for episode in sorted_episodes[:num_episodes]:
            print_episode_summary(episode, show_queries)
    
    if show_best:
        print(f"\n{'#'*80}")
        print(f"BEST {num_episodes} EPISODES (highest accuracy)")
        print(f"{'#'*80}")
        for episode in sorted_episodes[-num_episodes:]:
            print_episode_summary(episode, show_queries)
    
    if not show_best and not show_worst:
        # Show first N episodes by default
        print(f"\nShowing first {num_episodes} episodes:")
        for episode in episodes[:num_episodes]:
            print_episode_summary(episode, show_queries)
    
    # Overall statistics
    accuracies = [e['accuracy'] for e in episodes]
    print(f"\n{'='*80}")
    print("Overall Statistics:")
    print(f"{'='*80}")
    print(f"Mean accuracy: {sum(accuracies) / len(accuracies):.2%}")
    print(f"Min accuracy: {min(accuracies):.2%}")
    print(f"Max accuracy: {max(accuracies):.2%}")
    print(f"Perfect episodes (100% accuracy): {sum(1 for a in accuracies if a >= 0.9999)}")
    print(f"Failed episodes (<50% accuracy): {sum(1 for a in accuracies if a < 0.5)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze denoising text experiment episode details")
    parser.add_argument("result_path", type=str, help="Path to results JSON file")
    parser.add_argument("-n", "--num-episodes", type=int, default=5,
                        help="Number of episodes to show (default: 5)")
    parser.add_argument("--best", action="store_true",
                        help="Show best performing episodes")
    parser.add_argument("--worst", action="store_true",
                        help="Show worst performing episodes")
    parser.add_argument("--queries", action="store_true",
                        help="Show sample query strings for each episode")
    
    args = parser.parse_args()
    
    if not Path(args.result_path).exists():
        print(f"Error: File not found: {args.result_path}")
        return
    
    analyze_results(
        args.result_path,
        num_episodes=args.num_episodes,
        show_best=args.best,
        show_worst=args.worst,
        show_queries=args.queries
    )


if __name__ == "__main__":
    main()
