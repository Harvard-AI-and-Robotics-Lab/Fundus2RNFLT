"""Correct segmentation artifacts in every OCT RNFLT map of a dataset CSV with RNFLT2Vec inpainting
(TensorFlow 2.4 environment, see requirements-correction.txt). Writes
{out}/rnflt_maps_corrected/{eyeid}_corr.npy (one file per unique eye) and {out}/dataset_corr.csv
with the new rnflt_corr_path column."""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from fundus2rnflt.correction import DEFAULT_RNFLT2VEC_REPO, THRESHOLD_UM, blend_corrected, load_inpaint_predictor, predict_corrected_map


def parse_args():
    p = argparse.ArgumentParser(description="Inpaint RNFLT map artifacts with RNFLT2Vec")
    p.add_argument('--csv', type=str, required=True, help="dataset CSV with eyeid, rnflt_path and rnflt_mask_path columns")
    p.add_argument('--weights', type=str, required=True,
                   help="RNFLT2Vec weights (combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5; keep the file name)")
    p.add_argument('--output-dir', type=str, required=True, help="writes rnflt_maps_corrected/ and dataset_corr.csv under it")
    p.add_argument('--rnflt2vec-repo', type=str, default=DEFAULT_RNFLT2VEC_REPO, help="RNFLT2Vec code (the git submodule)")
    p.add_argument('--vgg16-weights', type=str, default='imagenet',
                   help="VGG16 weights RNFLT2Vec is built with ('imagenet' or a .h5); does not affect the output")
    p.add_argument('--threshold-um', type=float, default=THRESHOLD_UM, help="tissue thinner than this is inpainted (µm)")
    p.add_argument('--overwrite', action='store_true', help="Recompute maps whose file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    assert {'eyeid', 'rnflt_path', 'rnflt_mask_path'}.issubset(df.columns), "CSV needs eyeid, rnflt_path and rnflt_mask_path"
    assert 'rnflt_corr_path' not in df.columns, "CSV already has a rnflt_corr_path column"

    maps_dir = os.path.join(args.output_dir, 'rnflt_maps_corrected')
    os.makedirs(maps_dir, exist_ok=True)

    # One correction per unique eye (its OCT map); rows sharing an eye share the file.
    eyes = df[['eyeid', 'rnflt_path', 'rnflt_mask_path']].drop_duplicates().reset_index(drop=True)
    assert eyes['eyeid'].is_unique, "an eye maps to more than one RNFLT map"
    eyes['rnflt_corr_path'] = [os.path.join(maps_dir, f"{eyeid}_corr.npy") for eyeid in eyes['eyeid']]

    if args.overwrite:
        to_correct = eyes
    else:
        to_correct = eyes[[not os.path.exists(p) for p in eyes['rnflt_corr_path']]]
    print(f"{len(eyes)} unique eyes, {len(to_correct)} to correct")

    if len(to_correct) > 0:
        predictor = load_inpaint_predictor(args.weights, args.vgg16_weights, args.rnflt2vec_repo)
    for row in tqdm(to_correct.itertuples(index=False), total=len(to_correct), desc="Correcting RNFLT"):
        rnflt_um_224 = np.load(row.rnflt_path).astype(np.float32)
        mask_codes_224 = np.load(row.rnflt_mask_path).astype(np.uint8)
        pred_um_224 = predict_corrected_map(predictor, rnflt_um_224, mask_codes_224, args.threshold_um)
        rnflt_corr_224 = blend_corrected(rnflt_um_224, pred_um_224, mask_codes_224, args.threshold_um)
        np.save(row.rnflt_corr_path, rnflt_corr_224.astype(np.float32))

    df = df.merge(eyes[['eyeid', 'rnflt_corr_path']], on='eyeid', how='left', validate='many_to_one')
    assert df['rnflt_corr_path'].notna().all()
    output_csv = os.path.join(args.output_dir, 'dataset_corr.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV with rnflt_corr_path to {output_csv}")


if __name__ == '__main__':
    main()
