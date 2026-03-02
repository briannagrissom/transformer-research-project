# Performance Evaluation on Transformers Augmented with Hebbian and Gradient-Based Plasticity

**Authors:** Brianna Grissom & Christina Wang  
**Course:** APMTH 226 — Fall 2025

## Overview

This project evaluates decoder-only Transformers augmented with fast-weight components updated via neuromodulated **Hebbian** or **gradient-based** plasticity rules. The fast weights enable rapid, in-sequence adaptation at inference time without modifying the Transformer's static parameters.

We reproduce and extend the experiments from [Chaudhary (2025)](https://arxiv.org/abs/2504.07843), comparing three settings:

| Rule | Description |
|------|-------------|
| **None** | Standard Transformer (baseline) |
| **Hebbian** | Outer-product weight update gated by a learned neuromodulation signal |
| **Gradient** | Gradient descent on a local auxiliary loss to update fast weights |

## Experiments

1. **Copy Task** — short-term memory: recall *n* tokens after 20 distractor blanks.
2. **One-Shot Image Classification** — 5-way, 1-shot on CIFAR-FS and Omniglot.
3. **Extended Reproduction** — more training epochs + copying capacity vs. signal length.
4. **Three-Shot Text Category Classification** — 3-way, 3-shot word classification (animals / furniture / clothing).
5. **Pre-trained Transformer Baselines** — BERT and CLIP comparisons.

See `project.tex` for the full report.

## Repository Layout

```
.
├── project.tex                         # Final report (LaTeX source)
├── requirements.txt                    # Python dependencies
├── src/
│   ├── models/                         # Plastic Transformer, encoders
│   ├── tasks/                          # Task generators / datasets
│   └── experiments/                    # CLI entry points for each experiment
├── scripts/                            # Aggregation & plotting utilities
├── experiments/
│   ├── results/                        # All final result JSONs
│   └── tables/                         # Generated LaTeX tables
├── reproduction_figures/               # Figures for reproduced experiments
├── extended_figures/                   # Figures for extended experiments
└── word_category_figures/              # Figures for text classification experiment
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Run all commands from the repository root so Python resolves the `src` package.

## Running Experiments

Each experiment script accepts `--rule {none, hebbian, gradient}`, `--seeds`, `--output-path`, and `--device {auto, cpu, cuda, mps}`.

### Copy Task

```bash
python -m src.experiments.copying \
  --rule gradient --seq-length 5 --delay 20 --seeds 3 \
  --output-path experiments/results/copy_delay20_rule-gradient.json
```

### One-Shot Image Classification (CIFAR-FS / Omniglot)

```bash
python -m src.experiments.one_shot_classification \
  --rule hebbian --dataset cifarfs \
  --ways 5 --shots 1 --queries 15 \
  --epochs 20 --episodes-per-epoch 200 --seeds 3 \
  --output-path experiments/results/classification_cifarfs_rule-hebbian.json
```

Set `--dataset omniglot` for Omniglot. Torchvision downloads CIFAR-100 and Omniglot into `./data` automatically.

### Three-Shot Text Category Classification

```bash
python -m src.experiments.word_category \
  --rule gradient --seeds 3 \
  --output-path experiments/results/word_category_3shot_gradient_semantic-encoder_FINAL.json
```

### Copy Capacity (varying signal length)

```bash
python -m src.experiments.copy_capacity \
  --rule hebbian --seeds 3 \
  --output-path experiments/results/copy_capacity_rule-hebbian.json
```

## Generating Figures & Tables

```bash
python scripts/aggregate_results.py          # experiments/results/summary.json
python scripts/plot_results.py               # reproduction_figures/ & extended_figures/
python scripts/plot_word_category_results.py  # word_category_figures/
python scripts/build_tables.py               # experiments/tables/
```

## Key Findings

- **Reproduced results** match the original paper: Hebbian plasticity outperforms on one-shot image classification; all rules score similarly on the copy task.
- **With more training epochs**, the Gradient rule overtakes Hebbian on both image and text classification.
- **Copying capacity** degrades similarly for all three rules as signal length grows, but plasticity reduces variance across seeds.
- **Pre-trained BERT/CLIP** excel at their native modalities but fail the copy task (no in-context weight updates).

## References

- Chaudhary, S. (2025). *Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity.*
- Duan, Y. et al. (2023). *Hebbian and Gradient-based Plasticity Enables Robust Memory and Rapid Learning in RNNs.*
- Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners (GPT-3).*
