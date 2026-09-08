#!/usr/bin/env bash
# Fundus -> RNFLT U-Net, the paper's configuration (EfficientNet-B3 encoder, Mode A supervision on
# valid tissue of the original OCT map). Usage: bash scripts/train_rnflt.sh CSV OUT
# Encoder ablation (paper Table 3): replace --encoder-name with resnet34, resnet50 or efficientnet-b0.
set -eu
CSV="$1"
OUT="$2"
python scripts/train_rnflt.py --csv "$CSV" --output-dir "$OUT" \
    --encoder-name efficientnet-b3 --batch-size 64 --lr 1e-3 --weight-decay 5e-5 \
    --epochs 50 --patience 3 --loss mae --seed 42
