#!/bin/bash

# Compare pre-trained transformer baseline vs plastic transformer
# on word category classification

echo "========================================="
echo "Comparing Pre-trained vs Plastic Transformers"
echo "========================================="

# Test pre-trained BERT baseline
echo ""
echo "1. Testing Pre-trained BERT (frozen, prototype-based)"
echo "-----------------------------------------"
python -m src.experiments.pretrained_baseline \
    --model-name bert-base-uncased \
    --ways 3 \
    --shots 3 \
    --queries 2 \
    --train-episodes 200 \
    --test-episodes 100 \
    --output-path experiments/results/word_category_baseline_bert_3shot.json

echo ""
echo "========================================="
echo "Comparison completed!"
echo ""
echo "Results:"
echo "  - Pre-trained BERT: experiments/results/word_category_baseline_bert_3shot.json"
echo "  - Plastic (gradient): experiments/results/word_category_3shot_gradient_semantic-encoder_FINAL.json"
echo "  - Plastic (hebbian):  experiments/results/word_category_3shot_hebbian_semantic-encoder_FINAL.json"
echo "========================================="
