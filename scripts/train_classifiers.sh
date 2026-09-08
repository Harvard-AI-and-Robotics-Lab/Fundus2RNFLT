#!/usr/bin/env bash
# The seven glaucoma classifiers of the paper, with the hyperparameters of the reported runs.
# Usage: bash scripts/train_classifiers.sh CSV OUT   (CSV = dataset_pred.csv with pred_rnflt_path)
set -eu
CSV="$1"
OUT="$2"
COMMON=(--csv "$CSV" --output-dir "$OUT" --pretrained --rnflt-channels 1 --rnflt-norm zscore --batch-size 32 --epochs 30 --patience 5 --seed 42)

# fundus photograph only (class-weighted BCE, basic augmentation)
python scripts/train_classifier.py "${COMMON[@]}" --input-type fundus     --model-name resnet18 --lr 5e-4 --weight-decay 1e-5 --aug basic --class-weighted --amp
# OCT RNFLT map only
python scripts/train_classifier.py "${COMMON[@]}" --input-type rnflt_real --model-name resnet18 --lr 1e-4 --weight-decay 1e-4 --amp
# predicted RNFLT map only
python scripts/train_classifier.py "${COMMON[@]}" --input-type rnflt_pred --model-name resnet18 --lr 5e-4 --weight-decay 1e-4 --amp
# fundus + OCT RNFLT, mid-level concatenation fusion
python scripts/train_classifier.py "${COMMON[@]}" --input-type fused_real --model-name resnet18 --lr 1e-4 --weight-decay 1e-4 --amp
# fundus + predicted RNFLT, mid-level concatenation fusion
python scripts/train_classifier.py "${COMMON[@]}" --input-type fused_pred --model-name resnet18 --lr 1e-4 --weight-decay 1e-4 --amp
# fundus + OCT RNFLT, attention fusion
python scripts/train_classifier.py "${COMMON[@]}" --input-type fused_real --model-name resnet18_attn --lr 1e-4 --weight-decay 1e-4 \
    --attn-d-model 256 --attn-layers 1 --attn-heads 4 --attn-mlp-mult 2.0 --attn-dropout 0.1 --attn-beta-soft 1.5
# fundus + predicted RNFLT, attention fusion
python scripts/train_classifier.py "${COMMON[@]}" --input-type fused_pred --model-name resnet18_attn --lr 1e-4 --weight-decay 1e-4 \
    --attn-d-model 384 --attn-layers 2 --attn-heads 6 --attn-mlp-mult 2.0 --attn-dropout 0.05 --attn-beta-soft 1.5
