# Summary: Denoising Text Classification Implementation

## What Was Implemented

I've created a complete denoising text classification experiment that mirrors the structure of your one-shot image classification task. Here's what was built:

## New Files Created

### 1. **src/models/text_encoder.py** (Character-Level LSTM Encoder)
- Converts strings to fixed-size embeddings (replaces Conv4Encoder for images)
- Uses bidirectional LSTM to process character sequences
- Vocabulary: lowercase letters + space + special tokens ([PAD], [UNK])
- Output: 256-dimensional embeddings (configurable)

**Key Features:**
- Character embedding layer
- Bidirectional LSTM (captures context from both directions)
- Projection layer to match transformer input dimension

### 2. **src/tasks/text_noise.py** (Noise Generation Utilities)
- Two noise types as specified:
  - **Substitution**: Random character replacement (e.g., "hello" → "hellp")
  - **Transposition**: Adjacent character swaps (e.g., "hello" → "helol")
- Configurable noise probability `p` per character
- Edit distance computation (Levenshtein distance)
- Character error rate (CER) calculation

### 3. **src/tasks/text_classification.py** (Episode Sampling)
- 100-word vocabulary of common English words
- Train/test split: 80% train classes / 20% test classes (held-out)
- Episode sampling: 5-way, 1-shot by default
- Support set: **noisy** examples
- Query set: **clean** examples
- Fully configurable (ways, shots, queries, noise type, noise probability)

### 4. **src/experiments/denoising_text.py** (Main Experiment Script)
- Follows exact structure of `one_shot_classification.py`
- Training: 200 episodes per epoch with backprop
- Evaluation: 100 episodes on held-out classes (no backprop)
- Metrics:
  - Classification accuracy
  - Edit distance between noisy support and clean words
  - Confusion matrix
  - Plasticity diagnostics (η, ||δ||₂)
- Outputs JSON with all metrics and confusion matrix

### 5. **scripts/plot_denoising_results.py** (Visualization Utilities)
- Generates publication-quality plots:
  - Accuracy vs epoch curves
  - Training/validation loss curves
  - Confusion matrix heatmaps
  - Plasticity diagnostics (η and ||δ||₂)
  - Edit distance (character error rate) over time
  - Noise level comparison plots

### 6. **DENOISING_EXPERIMENT.md** (Complete Documentation)
- Full usage guide with examples
- Parameter explanations
- Example workflows
- Architecture overview

## How It Works (Step-by-Step)

### Episode Structure

Each episode follows this flow:

1. **Sample Episode**
   - Select 5 random words from vocabulary (e.g., "apple", "chair", "ocean", "guitar", "summer")
   - Generate 1 noisy support example per class
   - Generate 15 clean query examples per class

2. **Add Noise to Support Set**
   - Example with p=0.1 substitution:
     - "apple" → "appla"
     - "chair" → "chqir"
     - "ocean" → "ocdan"
     - etc.

3. **Encode Strings to Embeddings**
   - Support: ["appla", "chqir", ...] → CharacterEncoder → embeddings
   - Query: ["apple", "apple", ..., "chair", ...] → CharacterEncoder → embeddings

4. **Process Support Set** (Adapt Plastic Weights)
   ```python
   for noisy_embedding, label in support_set:
       input = [noisy_embedding, one_hot(label), 0]  # 0 = support marker
       transformer.forward_step(input, state)  # Updates plastic weights δ
   ```

5. **Process Query Set** (Compute Loss)
   ```python
   for clean_embedding, label in query_set:
       input = [clean_embedding, zeros, 1]  # 1 = query marker
       logits = transformer.forward_step(input, state)
       loss += cross_entropy(logits, label)
   ```

6. **Backpropagation** (Training Only)
   - Update static weights via gradient descent
   - Plastic weights reset each episode

### Train/Test Paradigm

- **Training**: Uses first 80 words of vocabulary (classes 0-79)
- **Evaluation**: Uses last 20 words of vocabulary (classes 80-99)
- This ensures the model truly generalizes to new classes (meta-learning)

## Key Differences from Image Classification

| Aspect | Image Classification | Text Classification |
|--------|---------------------|---------------------|
| Encoder | Conv4Encoder (CNN) | CharacterEncoder (LSTM) |
| Input | Images (tensors) | Strings (lists) |
| Preprocessing | None | Add character-level noise |
| Dataset | CIFAR-FS / Omniglot | 100-word vocabulary |
| Metrics | Accuracy only | Accuracy + Edit distance |

## Running the Experiment

### Quick Start (Single Run)

```bash
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-type substitution \
    --noise-prob 0.1 \
    --seeds 3 \
    --output-path experiments/results/denoising_test.json
```

### Full Experiment Specification

To replicate the experiment as described (all noise levels, both noise types):

```bash
# Substitution noise: p = 0.0, 0.05, 0.1, 0.2
for p in 0.0 0.05 0.1 0.2; do
    conda run -p ./.venv python -m src.experiments.denoising_text \
        --rule hebbian \
        --noise-type substitution \
        --noise-prob $p \
        --seeds 3 \
        --output-path experiments/results/denoising_sub_p${p}.json
done

# Transposition noise: p = 0.0, 0.05, 0.1, 0.2
for p in 0.0 0.05 0.1 0.2; do
    conda run -p ./.venv python -m src.experiments.denoising_text \
        --rule hebbian \
        --noise-type transposition \
        --noise-prob $p \
        --seeds 3 \
        --output-path experiments/results/denoising_trans_p${p}.json
done
```

### Visualize Results

```bash
# Install seaborn for plotting (if not already installed)
conda run -p ./.venv pip install seaborn

# Generate plots for a single experiment
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_sub_p0.1.json \
    --output-dir plots/denoising

# Compare across noise levels
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_sub_p0.1.json \
    --output-dir plots/denoising \
    --compare-noise \
        experiments/results/denoising_sub_p0.0.json \
        experiments/results/denoising_sub_p0.05.json \
        experiments/results/denoising_sub_p0.1.json \
        experiments/results/denoising_sub_p0.2.json
```

## Expected Runtime

Similar to copying task:
- **Per epoch**: ~1-2 minutes (200 episodes)
- **Full training**: ~20-40 minutes (20 epochs)
- **With 3 seeds**: ~1-2 hours

On Apple Silicon with MPS, expect faster performance.

## Output Metrics

The JSON output includes:

```json
{
  "aggregate": {
    "accuracy_mean": 0.84,
    "accuracy_std": 0.02,
    "loss_mean": 0.45,
    "edit_distance_mean": 0.12,
    "eta_mean": 0.15,
    "plastic_norm_mean": 0.8
  },
  "runs": [
    {
      "seed": 123,
      "final": {
        "accuracy": 0.85,
        "confusion_matrix": [[15, 0, 0, 0, 0], ...]
      }
    }
  ]
}
```

## Testing Before Running Full Experiment

You can test components individually:

```bash
# Test dataset and noise generation
conda run -p ./.venv python -m src.tasks.text_classification

# Test with minimal configuration (faster)
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --epochs 2 \
    --episodes-per-epoch 20 \
    --val-episodes 10 \
    --seeds 1
```

## Architecture Comparison

```
One-Shot Image Classification:
Image → Conv4Encoder → embedding → Plastic Transformer → logits

Denoising Text Classification:
String → CharacterEncoder (LSTM) → embedding → Plastic Transformer → logits
          ↑
      + noise (support only)
```

Both use the **same plastic transformer** - only the encoder changes!

## Next Steps

1. **Test the dataset**: Run `python -m src.tasks.text_classification` to see sample episodes
2. **Quick test run**: Run with `--epochs 2 --seeds 1` to verify everything works
3. **Full experiment**: Run all noise levels and types as shown above
4. **Visualize**: Generate plots with `plot_denoising_results.py`
5. **Compare rules**: Test `--rule hebbian`, `--rule gradient`, `--rule none`

## Key Parameters to Tune

- `--noise-prob`: Test at 0.0, 0.05, 0.1, 0.2 (as specified)
- `--noise-type`: Test both `substitution` and `transposition`
- `--rule`: Compare `hebbian` vs `gradient` vs `none`
- `--eta0`: Initial plasticity learning rate (default 0.2)
- `--model-dim`: Transformer dimension (default 256)

## Notes

- The implementation is fully compatible with your existing codebase
- Uses the same `PlasticTransformerModel` as image classification
- Train/test split ensures true meta-learning evaluation
- Confusion matrices are computed per episode (5×5 for 5-way classification)
- Edit distance measures the effectiveness of denoising

See `DENOISING_EXPERIMENT.md` for complete documentation!
