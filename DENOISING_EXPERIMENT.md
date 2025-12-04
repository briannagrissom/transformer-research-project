# Denoising Text Classification Experiment

This experiment tests the plastic transformer's ability to perform few-shot classification on noisy text inputs following a **support-noisy → query-clean** setup.

## Overview

The task is a 5-way, 1-shot classification where:
- **Support set**: Contains noisy examples (character substitution or transposition)
- **Query set**: Contains clean examples (no noise)
- **Vocabulary**: 100 common English words
- **Train/Test split**: 80% train classes / 20% test classes (held-out)

## Files Created

### Core Components

1. **`src/models/text_encoder.py`**
   - Character-level LSTM encoder
   - Converts variable-length strings to fixed-size embeddings
   - Bidirectional LSTM with projection layer

2. **`src/tasks/text_noise.py`**
   - Noise generation utilities
   - Two noise types:
     - `substitution`: Random character replacement
     - `transposition`: Adjacent character swaps
   - Edit distance computation for metrics

3. **`src/tasks/text_classification.py`**
   - Episode sampling for few-shot learning
   - 100-word vocabulary with train/test split
   - Generates noisy support and clean query examples

4. **`src/experiments/denoising_text.py`**
   - Main training/evaluation script
   - Follows same structure as `one_shot_classification.py`
   - Outputs JSON results with confusion matrix

5. **`scripts/plot_denoising_results.py`**
   - Visualization utilities
   - Generates plots: accuracy curves, loss, confusion matrices, plasticity diagnostics

## Installation

The experiment requires seaborn for plotting (optional):

```bash
conda run -p ./.venv pip install seaborn
```

## Running the Experiment

### Basic Usage

Run with default parameters (substitution noise, p=0.1):

```bash
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --output-path experiments/results/denoising_substitution_p0.1_hebbian.json
```

### Testing Different Noise Levels

The experiment specification requires testing at noise probabilities: 0.0, 0.05, 0.1, 0.2

```bash
# No noise (baseline)
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.0 \
    --output-path experiments/results/denoising_substitution_p0.0_hebbian.json

# Low noise
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.05 \
    --output-path experiments/results/denoising_substitution_p0.05_hebbian.json

# Medium noise
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_substitution_p0.1_hebbian.json

# High noise
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.2 \
    --output-path experiments/results/denoising_substitution_p0.2_hebbian.json
```

### Testing Different Noise Types

```bash
# Character substitution
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-type substitution \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_substitution_p0.1_hebbian.json

# Character transposition
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-type transposition \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_transposition_p0.1_hebbian.json
```

### Comparing Plasticity Rules

```bash
# Hebbian plasticity
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_p0.1_hebbian.json

# Gradient-based plasticity
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule gradient \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_p0.1_gradient.json

# No plasticity (baseline)
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule none \
    --noise-prob 0.1 \
    --output-path experiments/results/denoising_p0.1_none.json
```

## Key Parameters

### Task Configuration
- `--ways`: Number of classes per episode (default: 5)
- `--shots`: Support examples per class (default: 1)
- `--queries`: Query examples per class (default: 15)
- `--noise-type`: `substitution` or `transposition` (default: substitution)
- `--noise-prob`: Noise probability per character (default: 0.1)

### Model Configuration
- `--rule`: Plasticity rule - `none`, `hebbian`, or `gradient` (default: gradient)
- `--embedding-dim`: Text embedding dimension (default: 256)
- `--model-dim`: Transformer model dimension (default: 256)
- `--num-heads`: Attention heads (default: 4)
- `--num-layers`: Transformer layers (default: 2)
- `--eta0`: Initial plasticity learning rate (default: 0.2)

### Text Encoder Configuration
- `--char-embed-dim`: Character embedding dimension (default: 32)
- `--hidden-dim`: LSTM hidden dimension (default: 128)
- `--encoder-layers`: Number of LSTM layers (default: 2)

### Training Configuration
- `--epochs`: Number of training epochs (default: 20)
- `--episodes-per-epoch`: Episodes per epoch (default: 200)
- `--val-episodes`: Validation episodes (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--seeds`: Number of random seeds (default: 3)

### Device Configuration
- `--device`: `auto`, `cpu`, `cuda`, or `mps` (default: auto)

## Visualizing Results

### Single Experiment Visualization

Generate plots for a single experiment:

```bash
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_substitution_p0.1_hebbian.json \
    --output-dir plots/denoising
```

This generates:
- `*_accuracy.png`: Accuracy vs epoch
- `*_loss.png`: Training and validation loss vs epoch
- `*_confusion_matrix.png`: Confusion matrix heatmap
- `*_plasticity.png`: η(t) and ||δ||₂ diagnostics
- `*_edit_distance.png`: Character error rate vs epoch

### Comparing Noise Levels

Generate comparison plot across multiple noise levels:

```bash
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_substitution_p0.1_hebbian.json \
    --output-dir plots/denoising \
    --compare-noise \
        experiments/results/denoising_substitution_p0.0_hebbian.json \
        experiments/results/denoising_substitution_p0.05_hebbian.json \
        experiments/results/denoising_substitution_p0.1_hebbian.json \
        experiments/results/denoising_substitution_p0.2_hebbian.json
```

This generates:
- `noise_level_comparison.png`: Accuracy vs noise level with error bars

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "config": {
    "model": {...},
    "task": {...},
    "training": {...}
  },
  "runs": [
    {
      "seed": 123,
      "history": [...],
      "final": {
        "accuracy": 0.85,
        "loss": 0.42,
        "confusion_matrix": [[...], ...],
        "edit_distance_mean": 0.12,
        "eta_mean": 0.15,
        "plastic_norm_mean": 0.8,
        ...
      }
    },
    ...
  ],
  "aggregate": {
    "accuracy_mean": 0.84,
    "accuracy_std": 0.02,
    ...
  }
}
```

## Metrics Reported

1. **Classification Accuracy**: Average across query examples
2. **Edit Distance**: Character error rate between noisy support and clean words
3. **Confusion Matrix**: Per-class prediction breakdown
4. **Plasticity Diagnostics**:
   - η(t): Adaptive learning rate over time
   - ||δ||₂: Plastic weight norm over time

## Testing the Dataset

You can test the dataset and noise generation:

```bash
conda run -p ./.venv python -m src.tasks.text_classification
```

This will display:
- Sample episode with noisy support and clean query
- Noise examples at different probability levels

## Example Workflow

Complete workflow to replicate the experiment specification:

```bash
# Create results directory
mkdir -p experiments/results

# Run experiments for substitution noise at all levels
for p in 0.0 0.05 0.1 0.2; do
    conda run -p ./.venv python -m src.experiments.denoising_text \
        --rule hebbian \
        --noise-type substitution \
        --noise-prob $p \
        --seeds 3 \
        --output-path experiments/results/denoising_substitution_p${p}_hebbian.json
done

# Run experiments for transposition noise at all levels
for p in 0.0 0.05 0.1 0.2; do
    conda run -p ./.venv python -m src.experiments.denoising_text \
        --rule hebbian \
        --noise-type transposition \
        --noise-prob $p \
        --seeds 3 \
        --output-path experiments/results/denoising_transposition_p${p}_hebbian.json
done

# Generate visualizations and comparisons
mkdir -p plots/denoising

# Substitution noise comparison
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_substitution_p0.1_hebbian.json \
    --output-dir plots/denoising/substitution \
    --compare-noise \
        experiments/results/denoising_substitution_p0.0_hebbian.json \
        experiments/results/denoising_substitution_p0.05_hebbian.json \
        experiments/results/denoising_substitution_p0.1_hebbian.json \
        experiments/results/denoising_substitution_p0.2_hebbian.json

# Transposition noise comparison
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_transposition_p0.1_hebbian.json \
    --output-dir plots/denoising/transposition \
    --compare-noise \
        experiments/results/denoising_transposition_p0.0_hebbian.json \
        experiments/results/denoising_transposition_p0.05_hebbian.json \
        experiments/results/denoising_transposition_p0.1_hebbian.json \
        experiments/results/denoising_transposition_p0.2_hebbian.json
```

## Architecture Overview

The experiment follows the same structure as one-shot image classification:

1. **Text Encoding**: Strings → CharacterEncoder (LSTM) → embeddings
2. **Episode Structure**: 
   - Support: (noisy_embedding, one_hot_label, 0) → update plastic weights
   - Query: (clean_embedding, zeros, 1) → predict class
3. **Training**: Update static weights via backprop after each episode
4. **Evaluation**: Test on held-out classes (last 20% of vocabulary)

## Notes

- The vocabulary is split 80/20 for train/test to ensure true generalization
- Support examples are always noisy, query examples are always clean
- The confusion matrix is computed over the 5-way classification within each episode
- Edit distance measures the character-level corruption in support examples
- The model learns to denoise through the plastic weights' rapid adaptation
