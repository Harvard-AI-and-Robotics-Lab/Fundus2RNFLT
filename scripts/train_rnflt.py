"""Train the fundus -> RNFLT U-Net (Mode A: original maps on valid tissue; Mode B2: corrected maps
with down-weighted supervision). Writes {out}/logs/<model>/<run>/ and {out}/checkpoints/<model>/<run>/."""
import argparse
import datetime
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from fundus2rnflt.datasets import FundusToRNFLDataset
from fundus2rnflt.metrics import MaskedHuberLoss, masked_mae, masked_ssim_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for RNFLT prediction (original vs corrected)")
    parser.add_argument('--csv', type=str, required=True, help='dataset CSV with fundus_path, rnflt_path, rnflt_mask_path, split')
    parser.add_argument('--output-dir', type=str, required=True, help='run root; writes logs/ and checkpoints/ under it')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--model-name', type=str, default='unet-rnflt')
    parser.add_argument('--encoder-name', type=str, default='resnet34',
                        choices=['resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b3'])
    parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'huber'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)

    # Mode B2 switches (Mode A is unchanged when off)
    parser.add_argument('--use-corrected', action='store_true',
                        help='Train on corrected RNFLT with down-weighted supervision (Mode B2).')
    parser.add_argument('--corr-weight', type=float, default=0.3,
                        help='Down-weight for newly supervised pixels in corrected mode.')
    parser.add_argument('--thin-threshold', type=float, default=20.0,
                        help='µm threshold to mark originally-thin tissue in corrected mode.')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    # Run name: default appends the encoder and the supervision mode
    if args.model_name == 'unet-rnflt':
        suffix = 'corrected' if args.use_corrected else 'original'
        args.model_name = f'unet-rnflt-{args.encoder_name}-{suffix}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CKPT_DIR = os.path.join(args.output_dir, 'checkpoints', args.model_name, run_id)
    CHECKPOINT = os.path.join(CKPT_DIR, 'model.pth')
    LOG_DIR = os.path.join(args.output_dir, 'logs', args.model_name, run_id)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(f'{LOG_DIR}/sample_outputs', exist_ok=True)

    # Save config next to the logs and next to the checkpoint (predict_rnflt.py reads the latter)
    for config_dir in (LOG_DIR, CKPT_DIR):
        with open(os.path.join(config_dir, 'config.json'), 'w') as fp:
            json.dump(vars(args), fp, indent=2)

    tf_fundus = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr_rnflt = None  # dataset fills NaNs, no transform needed on rnflt

    # Mode A: use_corrected=False. Mode B2: use_corrected=True -> corrected target + weight map.
    train_ds = FundusToRNFLDataset(
        csv_path=args.csv,
        transform_fundus=tf_fundus,
        transform_rnflt=tr_rnflt,
        use_corrected=args.use_corrected,
        corr_weight=args.corr_weight,
        thin_threshold=args.thin_threshold,
    )
    val_ds = FundusToRNFLDataset(
        csv_path=args.csv,
        transform_fundus=tf_fundus,
        transform_rnflt=tr_rnflt,
        use_corrected=args.use_corrected,
        corr_weight=args.corr_weight,
        thin_threshold=args.thin_threshold,
    )

    train_ds.df = train_ds.df[train_ds.df['split'] == 'train'].reset_index(drop=True)
    val_ds.df = val_ds.df[val_ds.df['split'] == 'val'].reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Using device: {DEVICE}")
    print(f"Train samples: {len(train_ds)}    Val samples: {len(val_ds)}")
    print("Mode:", "corrected, down-weighted supervision" if args.use_corrected else "original, valid tissue only")

    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    if args.loss == 'mae':
        loss_fn = lambda pred, target, mask_or_weight: masked_mae(pred, target, mask_or_weight)
    elif args.loss == 'huber':
        loss_fn = MaskedHuberLoss(delta=1.0).to(DEVICE)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=3)

    best_val_loss = float('inf')
    no_improve = 0
    history = []

    print("Starting training loop…")
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.6f}")

        # -- train --
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="  Training", leave=False):
            x = batch['fundus'].to(DEVICE)          # Bx3xHxW
            y = batch['rnflt'].to(DEVICE)           # Bx1xHxW
            msk = batch['rnflt_mask'].to(DEVICE)    # Bx1xHxW in {0,1}

            pred = model(x)

            if args.use_corrected:
                # Mode B2: supervise with the down-weighted 'weight' map (disc/cup=0, newly filled <1)
                wmap = batch['weight'].to(DEVICE)   # Bx1xHxW
                batch_loss = loss_fn(pred, y, wmap)
            else:
                # Mode A: supervise on valid tissue only
                batch_loss = loss_fn(pred, y, msk)

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # -- validate -- (identical policy for both modes: valid tissue only)
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validation", leave=False):
                x = batch['fundus'].to(DEVICE)
                y = batch['rnflt'].to(DEVICE)
                msk = batch['rnflt_mask'].to(DEVICE)
                pred = model(x)
                batch_loss = loss_fn(pred, y, msk)
                running_val += batch_loss.item() * x.size(0)

        val_loss = running_val / len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })

        # checkpoint & early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT)
            print("  Saved best model.")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"-- Early stopping after {epoch} epochs (no improvement in {args.patience} epochs) --")
                break

    print("Training complete.")

    pd.DataFrame(history).to_csv(f'{LOG_DIR}/history.csv', index=False)
    print(f"Saved training history to {LOG_DIR}/history.csv")

    # ----------------------------
    # Evaluation on the validation set (metrics on valid tissue of the original map)
    # ----------------------------
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.eval()

    df = pd.read_csv(args.csv)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    mae_list, ssim_list = [], []

    # sample for qualitative triplets
    random.seed(args.seed)
    sample_idxs = set(random.sample(list(range(len(val_df))), min(5, len(val_df))))
    inv_norm = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                             std=[1 / s for s in [0.229, 0.224, 0.225]]),
    ])

    for i, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        x = tf_fundus(Image.open(row['fundus_path']).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()

        real = np.load(row['rnflt_path']).astype(np.float32)

        if 'rnflt_mask_path' in row and isinstance(row['rnflt_mask_path'], str) and os.path.exists(row['rnflt_mask_path']):
            mask_codes = np.load(row['rnflt_mask_path']).astype(np.uint8)
            valid = (mask_codes == 1)
        else:
            valid = np.isfinite(real)

        valid = valid & np.isfinite(real)
        if valid.sum() == 0:
            continue

        mae_list.append(np.abs(pred[valid] - real[valid]).mean())
        ssim_list.append(masked_ssim_numpy(pred, real, valid))

        if i in sample_idxs:
            fv = inv_norm(x.squeeze(0).cpu()).permute(1, 2, 0).numpy().clip(0, 1)
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(fv); axs[0].set_title('Fundus'); axs[0].axis('off')
            gt_vis = real.copy()
            gt_vis[~valid] = np.nan
            axs[1].imshow(np.ma.masked_invalid(gt_vis), cmap='jet'); axs[1].set_title('Target RNFLT (valid)'); axs[1].axis('off')
            axs[2].imshow(pred, cmap='jet'); axs[2].set_title('Predicted RNFLT'); axs[2].axis('off')
            fig.savefig(f'{LOG_DIR}/sample_outputs/triplet_{i}.png', bbox_inches='tight')
            plt.close(fig)

    metrics = {
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'num_samples': len(mae_list)
    }
    with open(f'{LOG_DIR}/metrics.json', 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved evaluation metrics to {LOG_DIR}/metrics.json")


if __name__ == '__main__':
    main()
