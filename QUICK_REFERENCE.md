# Quick Reference: Denoising Text Experiment

## Files Created
- `src/models/text_encoder.py` - Character LSTM encoder
- `src/tasks/text_noise.py` - Noise generation utilities  
- `src/tasks/text_classification.py` - Episode sampler
- `src/experiments/denoising_text.py` - Main experiment script
- `scripts/plot_denoising_results.py` - Visualization utilities
- `test_denoising.sh` - Quick test script

## Quick Commands

### Test the implementation
```bash
./test_denoising.sh
```

### Run single experiment
```bash
conda run -p ./.venv python -m src.experiments.denoising_text \
    --rule hebbian \
    --noise-prob 0.1 \
    --output-path experiments/results/test.json
```

### Run all noise levels (as specified)
```bash
for p in 0.0 0.05 0.1 0.2; do
    conda run -p ./.venv python -m src.experiments.denoising_text \
        --rule hebbian \
        --noise-type substitution \
        --noise-prob $p \
        --seeds 3 \
        --output-path experiments/results/denoising_sub_p${p}.json
done
```

### Visualize results
```bash
# Install seaborn first (if needed)
conda run -p ./.venv pip install seaborn

# Generate plots
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/test.json \
    --output-dir plots/denoising
```

### Compare noise levels
```bash
conda run -p ./.venv python scripts/plot_denoising_results.py \
    --result-path experiments/results/denoising_sub_p0.1.json \
    --output-dir plots/denoising \
    --compare-noise \
        experiments/results/denoising_sub_p0.0.json \
        experiments/results/denoising_sub_p0.05.json \
        experiments/results/denoising_sub_p0.1.json \
        experiments/results/denoising_sub_p0.2.json
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rule` | Plasticity rule: `none`, `hebbian`, `gradient` | `gradient` |
| `--noise-type` | `substitution` or `transposition` | `substitution` |
| `--noise-prob` | Noise probability (0.0 to 1.0) | `0.1` |
| `--ways` | Classes per episode | `5` |
| `--shots` | Support examples per class | `1` |
| `--queries` | Query examples per class | `15` |
| `--epochs` | Training epochs | `20` |
| `--seeds` | Random seeds to run | `3` |

## How It Works

1. **Sample episode**: 5 random words from vocabulary
2. **Add noise**: Apply character substitution/transposition to support examples
3. **Encode**: Convert strings to embeddings via character LSTM
4. **Support phase**: Feed noisy embeddings → adapt plastic weights
5. **Query phase**: Feed clean embeddings → predict class → compute loss
6. **Update**: Backprop to update static weights (plastic weights reset each episode)

## Expected Output

```json
{
  "aggregate": {
    "accuracy_mean": 0.84,
    "accuracy_std": 0.02,
    "edit_distance_mean": 0.12,
    "confusion_matrix": [[...], ...]
  }
}
```

## Documentation

- `DENOISING_EXPERIMENT.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- This file - Quick reference

## Differences from Image Classification

| Aspect | Images | Text |
|--------|--------|------|
| Encoder | Conv4Encoder | CharacterEncoder (LSTM) |
| Input | Tensors | Strings (lists) |
| Noise | None | Character substitution/transposition |
| Metrics | Accuracy | Accuracy + Edit distance |

Both use the **same PlasticTransformerModel**!
