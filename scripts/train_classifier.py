"""Train a glaucoma classifier on fundus, real RNFLT, predicted RNFLT, or a fusion of fundus and RNFLT.
Writes {out}/logs/glaucoma_classifier/<exp>/<run>/ and {out}/checkpoints/glaucoma_classifier/<exp>/<run>/."""
import argparse
import datetime
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from fundus2rnflt.datasets import GlaucomaDataset, compute_rnflt_stats
from fundus2rnflt.metrics import safe_auprc, safe_auroc
from fundus2rnflt.models import (RNFLTTransform, build_dual_branch_resnet, build_dual_branch_resnet_attn,
                                 build_single_stream_model, zero_rnflt_mask_channel_if_present)


def parse_args():
    p = argparse.ArgumentParser(description="Train glaucoma classifier")
    p.add_argument('--csv', type=str, required=True,
                   help='dataset CSV with fundus_path, rnflt_path, rnflt_mask_path, pred_rnflt_path, glaucoma_label, split')
    p.add_argument('--output-dir', type=str, required=True, help='run root; writes logs/ and checkpoints/ under it')

    p.add_argument('--input-type', type=str, required=True,
                   choices=['fundus', 'rnflt_real', 'rnflt_pred', 'fused_real', 'fused_pred'])
    p.add_argument('--model-name', type=str, default='resnet18',
                   choices=['resnet18', 'resnet50', 'mlp', 'resnet18_attn', 'resnet50_attn'])
    p.add_argument('--pretrained', action='store_true', help="Use ImageNet weights for CNN backbones")

    # RNFLT normalization & channels
    p.add_argument('--rnflt-channels', type=int, default=1,
                   help='RNFLT branch input channels (1 = map, 2 = map + valid-tissue mask)')
    p.add_argument('--rnflt-norm', type=str, default='zscore', choices=['zscore', 'minmax', 'none'])
    p.add_argument('--rnflt-mean', type=float, default=None,
                   help='RNFLT mean (z-score only); if None, computed from the train split')
    p.add_argument('--rnflt-std', type=float, default=None,
                   help='RNFLT std (z-score only); if None, computed from the train split')
    p.add_argument('--rnflt-min', type=float, default=0.0, help='RNFLT min for minmax scaling')
    p.add_argument('--rnflt-max', type=float, default=350.0, help='RNFLT max for minmax scaling')
    p.add_argument('--rnflt-fill', type=float, default=0.0, help='Value to fill NaNs before normalization')

    p.add_argument('--aug', type=str, default='none', choices=['none', 'basic'],
                   help="Augmentation mode (single-modality runs only; fusion runs use 'none')")

    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)

    p.add_argument('--rnflt-lr-scale', type=float, default=0.5,
                   help='Relative LR for the RNFLT branch vs the fundus branch (fusion only)')

    p.add_argument('--max-per-eye', type=int, default=0,
                   help='If >0, cap number of TRAIN samples per eye (reduces duplicate bias)')
    p.add_argument('--class-weighted', action='store_true', help='Use class-weighted BCE')

    p.add_argument('--amp', action='store_true', help='Use mixed precision (AMP)')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='Label smoothing epsilon for BCE targets (training only)')

    # attention fusion hyperparameters (only for *_attn)
    p.add_argument('--attn-d-model', type=int, default=384, help="Transformer token width")
    p.add_argument('--attn-layers', type=int, default=2, help="Number of fusion attention blocks")
    p.add_argument('--attn-heads', type=int, default=6, help="Number of attention heads")
    p.add_argument('--attn-mlp-mult', type=float, default=2.0, help="FFN expansion ratio in attention blocks")
    p.add_argument('--attn-dropout', type=float, default=0.1, help="Dropout in attention blocks and head")
    p.add_argument('--attn-beta-soft', type=float, default=1.5, help="Soft down-weight for newly supervised RNFLT tokens")
    return p.parse_args()


def cap_per_eye(df_train, max_per_eye):
    """Cap the number of TRAIN samples per eye to reduce duplicate bias."""
    if max_per_eye <= 0:
        return df_train.reset_index(drop=True)
    parts = []
    for eyeid, g in df_train.groupby('eyeid'):
        if len(g) > max_per_eye:
            parts.append(g.sample(max_per_eye, random_state=42))
        else:
            parts.append(g)
    capped = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return capped


def make_class_weights(df_train, label_col='glaucoma_label'):
    """pos_weight for BCEWithLogitsLoss (n_positive / n_negative, as the paper runs were trained)."""
    counts = df_train[label_col].value_counts()
    assert 0 in counts and 1 in counts, "class-weighted BCE needs both classes in the train split"
    w_pos = counts[0] / (counts[0] + counts[1])
    w_neg = counts[1] / (counts[0] + counts[1])
    pos_weight = torch.tensor(w_neg / max(w_pos, 1e-8), dtype=torch.float32)
    return pos_weight


def split_param_groups_for_fusion(model, base_lr, rnflt_lr_scale=0.5, weight_decay=1e-4):
    """Optimizer param groups: fundus branch (base_lr), RNFLT branch (scaled), head (base_lr)."""
    fundus_params, rnflt_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('fundus_enc'):
            fundus_params.append(p)
        elif n.startswith('rnflt_enc'):
            rnflt_params.append(p)
        else:
            head_params.append(p)
    return [
        {'params': fundus_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': rnflt_params, 'lr': base_lr * rnflt_lr_scale, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay},
    ]


def smooth_labels(y, eps):
    """y: Bx1 tensor with {0,1}; returns smoothed targets in [eps, 1-eps]."""
    if eps <= 0.0:
        return y
    return y * (1.0 - eps) + (1.0 - y) * eps


def batch_to_input(batch, device):
    parts = []
    if 'fundus' in batch:
        parts.append(batch['fundus'])
    if 'rnflt' in batch:
        parts.append(batch['rnflt'])
    return torch.cat(parts, dim=1).to(device) if len(parts) > 1 else parts[0].to(device)


def main():
    args = parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.input_type}-{args.model_name}"
    LOG_DIR = os.path.join(args.output_dir, 'logs', 'glaucoma_classifier', exp_name, run_id)
    CKPT_DIR = os.path.join(args.output_dir, 'checkpoints', 'glaucoma_classifier', exp_name, run_id)
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

    # Save config next to the logs and next to the checkpoint (evaluate.py reads the latter)
    for config_dir in (LOG_DIR, CKPT_DIR):
        with open(os.path.join(config_dir, 'config.json'), 'w') as fp:
            json.dump(vars(args), fp, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------
    # Transforms
    # -----------------------
    fundus_aug_mode = 'basic' if (args.input_type == 'fundus' and args.aug == 'basic') else 'none'
    rnflt_aug_mode = 'basic' if (args.input_type in ('rnflt_real', 'rnflt_pred') and args.aug == 'basic') else 'none'

    if fundus_aug_mode == 'none':
        tf_fundus = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        tf_fundus = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # Placeholder RNFLT transform; replaced below once z-score stats are known
    tr_rnflt = RNFLTTransform(norm=args.rnflt_norm,
                              mean=args.rnflt_mean, std=args.rnflt_std,
                              vmin=args.rnflt_min, vmax=args.rnflt_max,
                              fill_value=args.rnflt_fill, aug='none')

    # -----------------------
    # Datasets
    # -----------------------
    use_fundus = args.input_type in ('fundus', 'fused_real', 'fused_pred')

    if args.input_type == 'fundus':
        rnflt_source = 'none'
    elif args.input_type == 'rnflt_real':
        rnflt_source = 'real'
    elif args.input_type == 'rnflt_pred':
        rnflt_source = 'predicted'
    elif args.input_type == 'fused_real':
        rnflt_source = 'real'
    elif args.input_type == 'fused_pred':
        rnflt_source = 'predicted'
    else:
        raise ValueError(args.input_type)

    train_ds = GlaucomaDataset(args.csv, transform_fundus=tf_fundus,
                               transform_rnflt=tr_rnflt, use_fundus=use_fundus, rnflt_source=rnflt_source,
                               include_rnflt_mask_channel=(args.rnflt_channels == 2))
    val_ds = GlaucomaDataset(args.csv, transform_fundus=tf_fundus,
                             transform_rnflt=tr_rnflt, use_fundus=use_fundus, rnflt_source=rnflt_source,
                             include_rnflt_mask_channel=(args.rnflt_channels == 2))
    test_ds = GlaucomaDataset(args.csv, transform_fundus=tf_fundus,
                              transform_rnflt=tr_rnflt, use_fundus=use_fundus, rnflt_source=rnflt_source,
                              include_rnflt_mask_channel=(args.rnflt_channels == 2))

    train_ds.df = train_ds.df[train_ds.df['split'] == 'train'].reset_index(drop=True)
    val_ds.df = val_ds.df[val_ds.df['split'] == 'val'].reset_index(drop=True)
    test_ds.df = test_ds.df[test_ds.df['split'] == 'test'].reset_index(drop=True)

    if args.max_per_eye > 0:
        train_ds.df = cap_per_eye(train_ds.df, args.max_per_eye)

    # z-score stats from the TRAIN split unless given; saved for evaluation
    if rnflt_source != 'none' and args.rnflt_norm == 'zscore' and (args.rnflt_mean is None or args.rnflt_std is None):
        rnflt_key = 'rnflt_path' if rnflt_source == 'real' else 'pred_rnflt_path'
        mean, std = compute_rnflt_stats(train_ds.df, rnflt_key, fill_value=args.rnflt_fill)
        for stats_dir in (LOG_DIR, CKPT_DIR):
            with open(os.path.join(stats_dir, 'rnflt_norm_stats.json'), 'w') as fp:
                json.dump({'mean': mean, 'std': std, 'mode': 'zscore', 'fill': args.rnflt_fill}, fp, indent=2)

        # train gets optional augs ONLY for RNFLT-only runs; val/test never augmented
        train_tr_rnflt = RNFLTTransform(
            norm='zscore', mean=mean, std=std,
            vmin=args.rnflt_min, vmax=args.rnflt_max,
            fill_value=args.rnflt_fill,
            aug=rnflt_aug_mode
        )
        eval_tr_rnflt = RNFLTTransform(
            norm='zscore', mean=mean, std=std,
            vmin=args.rnflt_min, vmax=args.rnflt_max,
            fill_value=args.rnflt_fill,
            aug='none'
        )
        train_ds.tr = train_tr_rnflt
        val_ds.tr = eval_tr_rnflt
        test_ds.tr = eval_tr_rnflt

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Sanity checks consume one shuffled train batch each; kept because they advance the RNG state
    # that the paper runs were trained with.
    if args.input_type in ('rnflt_real', 'rnflt_pred'):
        for batch in train_loader:
            r = batch['rnflt']
            r_ch = batch['rnflt'].shape[1]
            assert r_ch == args.rnflt_channels, f"RNFLT channels={r_ch} != --rnflt-channels={args.rnflt_channels}"
            print(f"[Sanity] RNFLT per-channel min/max:",
                  [(float(r[:, i].min()), float(r[:, i].max())) for i in range(r_ch)])
            break

    if args.input_type in ('fused_real', 'fused_pred'):
        for b in train_loader:
            f = b['fundus']; r = b['rnflt']
            print("[Sanity] fundus shape:", tuple(f.shape), " rnflt shape:", tuple(r.shape))
            break

    print(f"Using device: {device}")
    print(f"Train/Val/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    # -----------------------
    # Build model
    # -----------------------
    if args.input_type in ('fused_real', 'fused_pred'):
        for batch in train_loader:
            if 'fundus' in batch and 'rnflt' in batch:
                f_ch = batch['fundus'].shape[1]
                r_ch = batch['rnflt'].shape[1]
                assert f_ch == 3, f"Expected fundus 3ch, got {f_ch}"
                assert r_ch == args.rnflt_channels, \
                    f"RNFLT channels mismatch: batch has {r_ch}, --rnflt-channels={args.rnflt_channels}"
            break

        if args.model_name in ('resnet18_attn', 'resnet50_attn'):
            backbone_name = 'resnet18' if args.model_name == 'resnet18_attn' else 'resnet50'
            model = build_dual_branch_resnet_attn(
                fundus_encoder=backbone_name,
                rnflt_encoder=backbone_name,
                rnflt_channels=args.rnflt_channels,
                pretrained_fundus=args.pretrained,
                pretrained_rnflt=False,
                d_model=args.attn_d_model,
                num_layers=args.attn_layers,
                num_heads=args.attn_heads,
                mlp_mult=args.attn_mlp_mult,
                dropout=args.attn_dropout,
                beta_soft=args.attn_beta_soft
            ).to(device)
        else:
            assert args.model_name in ('resnet18', 'resnet50'), \
                "Fusion baseline expects a ResNet encoder (resnet18/resnet50)."
            model = build_dual_branch_resnet(
                fundus_encoder=args.model_name,
                rnflt_encoder=args.model_name,
                rnflt_channels=args.rnflt_channels,
                pretrained_fundus=args.pretrained,
                pretrained_rnflt=False,
                dropout=0.3,
                hidden_dim=None
            ).to(device)

        zero_rnflt_mask_channel_if_present(model, expected_in_channels=args.rnflt_channels)

        param_groups = split_param_groups_for_fusion(
            model, base_lr=args.lr, rnflt_lr_scale=args.rnflt_lr_scale, weight_decay=args.weight_decay
        )
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    else:
        if args.input_type == 'fundus':
            in_ch = 3
        else:
            in_ch = args.rnflt_channels
        model = build_single_stream_model(args.model_name, in_ch, pretrained=args.pretrained).to(device)

        if args.input_type in ('rnflt_real', 'rnflt_pred') and args.rnflt_channels == 2:
            with torch.no_grad():
                w = model.conv1.weight  # [64, 2, 7, 7] for resnet18/50
                w[:, 1, :, :] = 0.0     # channel 1 is mask; channel 0 is rnflt

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.class_weighted:
        pos_weight = make_class_weights(train_ds.df, 'glaucoma_label').to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler(device='cuda', enabled=args.amp)

    # -----------------------
    # Training
    # -----------------------
    best_auc = 0.0
    no_imp = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            x = batch_to_input(batch, device)
            y = batch['label'].unsqueeze(1).float().to(device)

            y_smooth = smooth_labels(y, args.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y_smooth)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, ys, ps = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                x = batch_to_input(batch, device)
                y = batch['label'].unsqueeze(1).float().to(device)

                with autocast(device_type='cuda', enabled=args.amp):
                    logits = model(x)
                    val_losses.append(criterion(logits, y).item())
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                ys.extend(y.cpu().numpy())
                ps.extend(probs)

        y_true = np.asarray(ys).ravel()
        y_prob = np.asarray(ps).ravel()
        val_auc = safe_auroc(y_true, y_prob)
        val_pred = (y_prob >= 0.5).astype(int)
        val_acc = accuracy_score(y_true, val_pred)
        val_ap = safe_auprc(y_true, y_prob)

        lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)),
            'val_loss': float(np.mean(val_losses)),
            'val_auc': float(val_auc),
            'val_acc': float(val_acc),
            'val_ap': float(val_ap),
            'lr': float(lr)
        })
        print(f"Epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={history[-1]['val_loss']:.4f} "
              f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} val_ap={val_ap:.4f} lr={lr:.6f}")

        # Early stopping on AUC
        if np.isnan(val_auc):
            no_imp += 1
        elif val_auc > best_auc:
            best_auc = val_auc
            no_imp = 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/model.pth")
        else:
            no_imp += 1

        if no_imp >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(f"{LOG_DIR}/history.csv", index=False)

    # -----------------------
    # Test (best-val checkpoint)
    # -----------------------
    model.load_state_dict(torch.load(f"{CKPT_DIR}/model.pth", map_location=device, weights_only=True))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            x = batch_to_input(batch, device)
            y = batch['label'].unsqueeze(1).float().to(device)

            with autocast(device_type='cuda', enabled=args.amp):
                probs = torch.sigmoid(model(x)).detach().cpu().numpy()
            ys.extend(y.cpu().numpy()); ps.extend(probs)

    y_true = np.asarray(ys).ravel()
    y_prob = np.asarray(ps).ravel()
    test_auc = safe_auroc(y_true, y_prob)
    test_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    test_ap = safe_auprc(y_true, y_prob)

    metrics = {
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_ap': float(test_ap),
        'best_val_auc': float(best_auc)
    }
    with open(f"{LOG_DIR}/metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}, Test AP: {test_ap:.4f}")


if __name__ == '__main__':
    main()
