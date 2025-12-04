# Episode Details Analysis - Denoising Text Experiment

## What Changed

The `denoising_text.py` experiment now tracks detailed information about each evaluation episode, including:
- Support strings (noisy examples used for adaptation)
- Class words (the clean words for each class)
- Query predictions and true labels
- Per-episode accuracy and loss

This allows you to see exactly which noisy strings were used and how they affected the model's predictions.

## Output Format

The results JSON now includes an `episode_details` list in the evaluation metrics:

```json
{
  "final": {
    "accuracy": 0.85,
    "loss": 0.42,
    "confusion_matrix": [[...], ...],
    "episode_details": [
      {
        "episode_idx": 0,
        "support_strings": ["appla", "chqir", "ocdan", "guitbr", "summfr"],
        "support_labels": [0, 1, 2, 3, 4],
        "class_words": ["apple", "chair", "ocean", "guitar", "summer"],
        "query_predictions": [0, 0, 1, 1, 2, ...],
        "query_true_labels": [0, 0, 1, 1, 2, ...],
        "query_strings": ["apple", "apple", "chair", ...],
        "accuracy": 0.87,
        "loss": 0.38
      },
      ...
    ]
  }
}
```

## Using the Analysis Script

### Basic Usage

Show first 5 episodes:
```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json
```

### Show Best Performing Episodes

```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --best -n 10
```

### Show Worst Performing Episodes

```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --worst -n 10
```

### Include Query String Details

```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --best --queries -n 5
```

### Show Both Best and Worst

```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --best --worst -n 3
```

## Example Output

```
================================================================================
Episode 42
================================================================================
Accuracy: 93.33%
Loss: 0.2341

Classes and Support Strings (noisy):
  Class 0: 'apple'
    Support: 'appla' (noisy)
  Class 1: 'chair'
    Support: 'chqir' (noisy)
  Class 2: 'ocean'
    Support: 'ocean' (no noise)
  Class 3: 'guitar'
    Support: 'guitbr' (noisy)
  Class 4: 'summer'
    Support: 'summfr' (noisy)

Per-Class Query Accuracy:
  'apple' (class 0): 15/15 = 100.00%
  'chair' (class 1): 14/15 = 93.33%
  'ocean' (class 2): 15/15 = 100.00%
  'guitar' (class 3): 12/15 = 80.00%
  'summer' (class 4): 14/15 = 93.33%

Sample Query Strings (first 10):
    ✓ 'apple' → predicted: 'apple', true: 'apple'
    ✓ 'apple' → predicted: 'apple', true: 'apple'
    ✗ 'chair' → predicted: 'apple', true: 'chair'
    ✓ 'chair' → predicted: 'chair', true: 'chair'
    ...
```

## What You Can Learn

### 1. Impact of Noise on Specific Words

See which words are more robust to noise:
```bash
# Look at worst episodes to see which support strings caused problems
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --worst -n 20 | grep "Support:"
```

### 2. Confusion Patterns

Identify which words get confused with each other:
```bash
# Show query details to see prediction errors
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --worst --queries -n 5
```

### 3. Best vs Worst Episodes

Compare what makes an episode easy or hard:
```bash
conda run -p ./.venv python scripts/analyze_episodes.py \
    experiments/results/denoising_sub_p0.1_hebbian.json \
    --best --worst -n 5
```

## Programmatic Analysis

You can also load and analyze the episode details programmatically:

```python
import json

# Load results
with open('experiments/results/denoising_sub_p0.1_hebbian.json', 'r') as f:
    data = json.load(f)

# Get episode details
episodes = data['runs'][-1]['final']['episode_details']

# Analyze specific patterns
for episode in episodes:
    support_strings = episode['support_strings']
    class_words = episode['class_words']
    accuracy = episode['accuracy']
    
    # Check if any support string had no noise
    no_noise_count = sum(1 for s, w in zip(support_strings, class_words) if s == w)
    
    print(f"Episode {episode['episode_idx']}: "
          f"accuracy={accuracy:.2%}, "
          f"no_noise_support={no_noise_count}")
```

## Notes

- Query strings are limited to the first 10 per episode to keep output manageable
- All 75 query predictions/labels are still stored for analysis
- Episode details are only stored for the final evaluation (not during training)
- The feature works with all noise types (substitution and transposition)

## Backward Compatibility

Old result files (without episode details) will still work with other analysis scripts. The `analyze_episodes.py` script will detect missing episode details and show an appropriate message.
