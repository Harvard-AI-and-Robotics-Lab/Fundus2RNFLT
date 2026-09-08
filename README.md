# Fundus2RNFLT

The code for the paper [**Deriving OCT-Equivalent Retinal Nerve Fiber Layer Thickness Maps from Fundus Photographs with Deep Learning Improves Glaucoma Diagnosis**](https://doi.org/10.1016/j.xops.2026.101325) published in Ophthalmology Science. Lily Shi, Min Shi, In Young Chung, Louis R. Pasquale, Lucy Q. Shen, Mengyu Wang. 

## Abstract

**Purpose:** To develop and evaluate a deep learning model that predicts OCT-equivalent retinal nerve fiber layer thickness (RNFLT) maps directly from color fundus photographs and to assess their diagnostic value for detecting glaucomatous visual field (VF) loss.

**Design:** Retrospective model development and evaluation study.

**Participants:** 15 031 paired fundus photographs and spectral-domain OCT scans collected at Massachusetts Eye and Ear, 2011 to 2022.

**Methods:** Paired fundus and OCT images were used to train a U-Net-based model to predict pixel-wise RNFLT maps with artifact-corrected supervision. Diagnostic performance was evaluated across single-modality models (fundus photos only, real RNFLT maps, and predicted RNFLT maps) and multimodal fusion models (fundus + predicted RNFLT maps). Stratified analyses examined model performance across glaucoma severity and demographic subgroups. Glaucoma was defined based on standard criteria applied to Humphrey 24-2 VF testing.

**Main Outcome Measures:** Mean absolute error and structural similarity index for RNFLT map prediction. Area under the receiver operating characteristic curve (AUC) and accuracy for glaucoma detection.

**Results:** RNFLT map prediction achieved mean absolute error = 15.4 μm and structural similarity index = 0.65, measured against artifact-corrected RNFLT maps derived from OCT. For glaucoma detection, the predicted RNFLT-only classifier outperformed the fundus-only classifier (AUC: 0.889 vs. 0.883, P < 0.005; accuracy 82.0% vs. 78.0%), but performed worse than the real-RNFLT-only classifier (AUC: 0.889 vs. 0.903, P < 0.005). Multimodal fusion of fundus images with predicted RNFLT maps improved performance, achieving an AUC of 0.909, outperforming all single-modality inputs (P < 0.005 vs. fundus-only, predicted-RNFLT-only, and real-RNFLT-only). Performance gains from fundus-only to multimodal classifier were greater in early-stage glaucoma compared to severe cases: accuracy increased from 55.3% to 64.0% in mild cases, from 71.5% to 80.4% in moderate cases, and from 90.0% to 94.6% in severe cases.

**Conclusions:** Predicted RNFLT maps derived from fundus photographs provide quantitative, OCT-like structural information and improve detection of glaucomatous VF loss. Unlike prior work that predicted only summary RNFLT values, our model generates full RNFLT maps that better support glaucoma classification than fundus images alone. This approach offers a scalable pathway for early glaucoma screening and expands diagnostic access in resource-limited settings.

## Dataset

The study data (Zeiss Visucam fundus photographs, Cirrus OCT scans and Humphrey 24-2 visual fields from Massachusetts Eye and Ear, 2011-2022) contain protected health information and cannot be released. The code runs on any data organised as below.

### Expected input layout (`--data-dir`)

```
data/
├── fundus.csv                 # one row per fundus photograph
├── oct.csv                    # one row per OCT scan
├── vf.csv                     # one row per visual-field test
├── fundus_images/<jpgfile>    # referenced by fundus.csv
└── oct_scans/<datadir>/       # referenced by oct.csv, one directory per scan
    ├── segmentation_ilm.csv           # ILM surface,  200 x 200 values, one per line
    ├── segmentation_rnfl_to_gcl.csv   # RNFL/GCL surface, same layout
    ├── mask_disc.csv                  # 0/1 optic disc mask (optional, zeros if absent)
    └── mask_cup.csv                   # 0/1 optic cup mask (optional, zeros if absent)
```

## Requirements

Two Python environments are needed. The artifact-correction stage depends on RNFLT2Vec, which requires TensorFlow 2.4; everything else runs on PyTorch.

| Environment | Python | Used by | Install |
|---|---|---|---|
| PyTorch | 3.10 | `preprocess.py`, `train_rnflt.py`, `predict_rnflt.py`, `train_classifier.py`, `evaluate.py`, `analyze_rnflt_maps.py` | `pip install -r requirements.txt && pip install -e .` |
| TensorFlow 2.4 | 3.8 | `correct_rnflt.py` | `pip install -r requirements-correction.txt && pip install -e .` |

```bash
git clone --recursive https://github.com/Harvard-AI-and-Robotics-Lab/Fundus2RNFLT   # pulls third_party/RNFLT2Vec
cd Fundus2RNFLT
# in each environment:
pip install -r requirements.txt && pip install -e .                # PyTorch environment
pip install -r requirements-correction.txt && pip install -e .     # TensorFlow 2.4 environment
```

RNFLT2Vec is a git submodule at `third_party/RNFLT2Vec` pointing to [Harvard-AI-and-Robotics-Lab/RNFLT2Vec](https://github.com/Harvard-AI-and-Robotics-Lab/RNFLT2Vec). If you cloned without `--recursive`, run `git submodule update --init`. The correction stage runs on CPU (about 15 eyes per second); no GPU is needed for it. The PyTorch stages expect a CUDA GPU.

Download the weights bundle (see [Pretrained Model](#pretrained-model)) into `weights/`. Keep the RNFLT2Vec file name `combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5`: RNFLT2Vec parses the training epoch from it. With the default `--vgg16-weights imagenet`, Keras downloads the 58 MB ImageNet VGG16 file once into `~/.keras/models/`; pass any local VGG16 `.h5` to avoid the download. VGG16 only enters RNFLT2Vec's training loss, so the corrected maps do not depend on this choice.

## Experiments

The six stages below reproduce the paper end to end. 

```bash
# 1. raw tables -> master table -> RNFLT maps + masks -> QC filter -> patient split
python scripts/preprocess.py --data-dir data/ --output-dir data/processed/

# 2. artifact correction with RNFLT2Vec (TensorFlow 2.4 environment, CPU)
python scripts/correct_rnflt.py --csv data/processed/dataset.csv \
    --weights weights/combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5 --output-dir data/processed/

# 3. fundus -> RNFLT U-Net (EfficientNet-B3 encoder), then predicted maps for every row
bash scripts/train_rnflt.sh data/processed/dataset_corr.csv runs/
python scripts/predict_rnflt.py --csv data/processed/dataset_corr.csv \
    --checkpoint runs/checkpoints/unet-rnflt-efficientnet-b3/<run>/model.pth --output-dir data/processed/

# 4. the seven glaucoma classifiers of the paper
bash scripts/train_classifiers.sh data/processed/dataset_pred.csv runs/

# 5. evaluation: per-run metrics, leaderboard, all pairwise DeLong and bootstrap comparisons
bash scripts/evaluate.sh data/processed/dataset_pred.csv runs/checkpoints/glaucoma_classifier/ results/

# 6. predicted-map analysis: MAE by severity / thickness / subgroup, Garway-Heath sectors,
#    structure-function correlations, circumpapillary RNFLT AUC
python scripts/analyze_rnflt_maps.py --csv data/processed/dataset_pred.csv --data-dir data/ --output-dir results/maps/
```

To skip training, point stage 3 at `weights/unet/unet-rnflt-efficientnet-b3/<run>/model.pth` and stage 5 at `weights/classifiers/`.

### Runs in the paper

`scripts/train_classifiers.sh` trains the seven classifiers of Table 8:

| Line in `train_classifiers.sh` | `--input-type` | `--model-name` | Table 8 row |
|---|---|---|---|
| 1 | `fundus` | `resnet18` | fundus photograph only |
| 2 | `rnflt_real` | `resnet18` | OCT RNFLT map only |
| 3 | `rnflt_pred` | `resnet18` | predicted RNFLT map only |
| 4 | `fused_real` | `resnet18` | fundus + OCT RNFLT, concatenation fusion |
| 5 | `fused_pred` | `resnet18` | fundus + predicted RNFLT, concatenation fusion |
| 6 | `fused_real` | `resnet18_attn` | fundus + OCT RNFLT, attention fusion (d_model 256, 1 layer, 4 heads) |
| 7 | `fused_pred` | `resnet18_attn` | fundus + predicted RNFLT, attention fusion (d_model 384, 2 layers, 6 heads) |

`scripts/train_rnflt.sh` trains the U-Net of the paper (EfficientNet-B3 encoder, batch 64, learning rate 1e-3, MAE loss). The encoder comparison of Table 3 is obtained by changing `--encoder-name` to `resnet34`, `resnet50` or `efficientnet-b0`.

## Pretrained Model

The checkpoints of the runs reported in the paper and the RNFLT2Vec weights used for artifact correction are distributed as one bundle: **TODO: https://huggingface.co/harvardairobotics/Fundus2RNFLT**. Unpack it into `weights/`.

| Path in `weights/` | Model | Paper |
|---|---|---|
| `unet/unet-rnflt-efficientnet-b3/20250824_195258/` | fundus -> RNFLT U-Net, EfficientNet-B3 encoder | Table 3 (B3 row); source of every predicted map |
| `classifiers/fundus-resnet18/20250830_184641/` | fundus only | Table 8 |
| `classifiers/rnflt_real-resnet18/20250831_201332/` | OCT RNFLT only | Table 8 |
| `classifiers/rnflt_pred-resnet18/20250907_172539/` | predicted RNFLT only | Table 8 |
| `classifiers/fused_real-resnet18/20250907_120946/` | fundus + OCT RNFLT, concatenation | Table 8 |
| `classifiers/fused_pred-resnet18/20250913_134801/` | fundus + predicted RNFLT, concatenation | Table 8 |
| `classifiers/fused_real-resnet18_attn/20251022_221431/` | fundus + OCT RNFLT, attention | Table 8 |
| `classifiers/fused_pred-resnet18_attn/20251025_162943/` | fundus + predicted RNFLT, attention | Table 8 |
| `combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5` | RNFLT2Vec (artifact correction) | Methods |

```bash
python scripts/predict_rnflt.py --csv data/processed/dataset_corr.csv \
    --checkpoint weights/unet/unet-rnflt-efficientnet-b3/20250824_195258/model.pth --output-dir data/processed/
bash scripts/evaluate.sh data/processed/dataset_pred.csv weights/classifiers/ results/
```

## Acknowledgment and Citation

Artifact correction uses [RNFLT2Vec](https://github.com/Harvard-AI-and-Robotics-Lab/RNFLT2Vec), the successor of [EyeLearn](https://github.com/Harvard-AI-and-Robotics-Lab/EyeLearn), included as a git submodule. If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{shi2026fundus2rnflt,
  title={Deriving OCT-Equivalent Retinal Nerve Fiber Layer Thickness Maps from Fundus Photographs with Deep Learning Improves Glaucoma Diagnosis},
  author={Shi, Lily and Shi, Min and Chung, In Young and Pasquale, Louis R. and Shen, Lucy Q. and Wang, Mengyu},
  journal={Ophthalmology Science},
  volume={6},
  number={10},
  pages={101325},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.xops.2026.101325}
}
```
