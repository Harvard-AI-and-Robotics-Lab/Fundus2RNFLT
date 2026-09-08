"""Predict an RNFLT map from every fundus photograph in a dataset CSV with a trained U-Net.
Writes {out}/rnflt_maps_pred/<model>/<run>/{eyeid}__{fundus_stem}.npy (one file per unique
(eyeid, fundus_path)) and {out}/dataset_pred.csv with the new pred_rnflt_path column."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

INPUT_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description="Generate predicted RNFLT maps from fundus photographs")
    p.add_argument('--csv', type=str, required=True, help="dataset CSV with eyeid and fundus_path columns")
    p.add_argument('--checkpoint', type=str, required=True,
                   help="U-Net model.pth; its config.json (encoder_name) must sit next to it")
    p.add_argument('--output-dir', type=str, required=True,
                   help="writes rnflt_maps_pred/<model>/<run>/ and dataset_pred.csv under it")
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--clip-min', type=float, default=0.0, help="Min clip for RNFLT predictions (µm)")
    p.add_argument('--clip-max', type=float, default=350.0, help="Max clip for RNFLT predictions (µm)")
    p.add_argument('--overwrite', action='store_true', help="Recompute maps whose file already exists")
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv(args.csv)
    assert 'pred_rnflt_path' not in df.columns, "CSV already has a pred_rnflt_path column"

    checkpoint_path = Path(args.checkpoint)
    with open(checkpoint_path.parent / 'config.json') as fp:
        config = json.load(fp)
    run_id = checkpoint_path.parent.name
    model_name = checkpoint_path.parents[1].name
    maps_dir = os.path.join(args.output_dir, 'rnflt_maps_pred', model_name, run_id)
    os.makedirs(maps_dir, exist_ok=True)

    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    tf_fundus = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # One prediction per unique (eyeid, fundus image); rows sharing an image share the file.
    unique_images = df[['eyeid', 'fundus_path']].drop_duplicates().reset_index(drop=True)
    unique_images['pred_rnflt_path'] = [
        os.path.join(maps_dir, f"{eyeid}__{Path(fundus_path).stem}.npy")
        for eyeid, fundus_path in zip(unique_images['eyeid'], unique_images['fundus_path'])
    ]
    assert unique_images['pred_rnflt_path'].is_unique, "Two different (eyeid, fundus image) pairs map to the same file name"

    if args.overwrite:
        to_predict = unique_images
    else:
        to_predict = unique_images[[not os.path.exists(p) for p in unique_images['pred_rnflt_path']]]
    print(f"{len(unique_images)} unique images, {len(to_predict)} to predict")

    with torch.no_grad():
        for start in tqdm(range(0, len(to_predict), args.batch_size), desc="Predicting RNFLT"):
            rows = to_predict.iloc[start:start + args.batch_size]
            batch = torch.stack([tf_fundus(Image.open(p).convert('RGB')) for p in rows['fundus_path']], dim=0).to(device)
            pred = model(batch).squeeze(1).cpu().numpy()  # [B,H,W]
            for k, out_path in enumerate(rows['pred_rnflt_path']):
                arr = pred[k].astype(np.float32)
                arr = np.clip(arr, args.clip_min, args.clip_max, out=arr)
                np.save(out_path, arr)

    df = df.merge(unique_images, on=['eyeid', 'fundus_path'], how='left', validate='many_to_one')
    assert df['pred_rnflt_path'].notna().all()
    output_csv = os.path.join(args.output_dir, 'dataset_pred.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV with pred_rnflt_path to {output_csv}")


if __name__ == '__main__':
    main()
