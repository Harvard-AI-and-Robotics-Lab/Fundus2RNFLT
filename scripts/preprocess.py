"""Raw hospital tables -> master table -> RNFLT maps + masks -> QC filter -> patient-level split.

--data-dir layout: fundus.csv, oct.csv, vf.csv, fundus_images/<jpgfile>, oct_scans/<datadir>/
Writes under --output-dir: master_table.csv, rnflt_maps/<eyeid>.npy + <eyeid>_mask.npy,
qc.csv, dropped.csv, dataset.csv (QC-filtered rows with patient_id and split)."""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from fundus2rnflt.master_table import assign_glaucoma_label, build_master_table, select_core_columns
from fundus2rnflt.rnflt_maps import (assign_patient_split, build_rnflt_map_and_mask, compute_qc_for_eye,
                                     drop_mask, has_segmentation)

DATASET_COLUMNS = ["eyeid", "fundus_path", "rnflt_path", "rnflt_mask_path", "glaucoma_label",
                   "age", "race", "male", "ethnicity", "vfi", "md", "psdprob",
                   "invalid_tissue_pct", "gt300_pct_valid", "valid_pct", "max_um", "p95_um", "p99_um",
                   "disc_pct", "cup_pct"]


def parse_args():
    p = argparse.ArgumentParser(description="Build the master table, RNFLT maps, QC filter and patient split")
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=42, help='patient split seed')
    p.add_argument('--thresh-invalid', type=float, default=20.0, help='drop eyes with > this %% invalid tissue')
    p.add_argument('--thresh-extreme', type=float, default=2.0, help='drop eyes with > this %% of valid pixels above 300 µm')
    return p.parse_args()


def build_maps(master, data_dir, maps_dir):
    """One RNFLT map + mask per eye, built from the first master row of that eye; existing files are
    kept. Returns one record per master row whose eye has a map (rows whose OCT folder has no
    segmentation are skipped, as in the original pipeline)."""
    os.makedirs(maps_dir, exist_ok=True)
    records = []
    n_built = 0
    n_no_segmentation = 0
    for _, row in tqdm(master.iterrows(), total=len(master), desc="RNFLT maps"):
        eyeid = f"{row['id']}-{row['righteye']}"
        fundus_path = os.path.join(data_dir, 'fundus_images', row['jpgfile'])
        oct_dir = os.path.join(data_dir, 'oct_scans', row['datadir'])
        rnflt_path = os.path.join(maps_dir, f"{eyeid}.npy")
        mask_path = os.path.join(maps_dir, f"{eyeid}_mask.npy")

        if not (os.path.exists(rnflt_path) and os.path.exists(mask_path)):
            if not has_segmentation(oct_dir):
                n_no_segmentation += 1
                continue
            rnflt_map, mask_codes = build_rnflt_map_and_mask(oct_dir)
            np.save(rnflt_path, rnflt_map.astype(np.float32))
            np.save(mask_path, mask_codes.astype(np.uint8))
            n_built += 1

        records.append({
            'eyeid': eyeid,
            'fundus_path': fundus_path,
            'rnflt_path': rnflt_path,
            'rnflt_mask_path': mask_path,
            'glaucoma_label': row['glaucoma_label'],
            'age': row['age'],
            'race': row['race'],
            'male': row['male'],
            'ethnicity': row['hispanic'],
            'vfi': row['vfi'],
            'md': row['md'],
            'psdprob': row['psdprob'],
        })
    print(f"Rows with maps: {len(records)}; maps built now: {n_built}; rows skipped (no segmentation): {n_no_segmentation}")
    return pd.DataFrame(records)


def compute_qc_per_eye(unfiltered):
    eyes = unfiltered.drop_duplicates(subset='eyeid')[['eyeid', 'rnflt_path', 'rnflt_mask_path']]
    qc_records = []
    for _, row in tqdm(eyes.iterrows(), total=len(eyes), desc="QC"):
        qc = compute_qc_for_eye(row['rnflt_path'], row['rnflt_mask_path'])
        qc_records.append({'eyeid': row['eyeid'], **qc})
    return pd.DataFrame(qc_records)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fundus = pd.read_csv(os.path.join(args.data_dir, 'fundus.csv'))
    oct_ = pd.read_csv(os.path.join(args.data_dir, 'oct.csv'))
    vf = pd.read_csv(os.path.join(args.data_dir, 'vf.csv'))
    master = select_core_columns(assign_glaucoma_label(build_master_table(fundus, oct_, vf)))
    n_unlabelled = int(master['glaucoma_label'].isna().sum())
    assert n_unlabelled == 0, f"{n_unlabelled} rows satisfy neither the glaucoma nor the normal label rule"
    master.to_csv(os.path.join(args.output_dir, 'master_table.csv'), index=False)
    print(f"Master table: {len(master)} rows")

    unfiltered = build_maps(master, args.data_dir, os.path.join(args.output_dir, 'rnflt_maps'))

    qc = compute_qc_per_eye(unfiltered)
    qc.to_csv(os.path.join(args.output_dir, 'qc.csv'), index=False)

    merged = unfiltered.merge(qc, on='eyeid', how='inner', validate='many_to_one')
    drop, reasons = drop_mask(merged, thresh_invalid=args.thresh_invalid, thresh_extreme=args.thresh_extreme)

    dropped = merged[drop][DATASET_COLUMNS].copy()
    dropped['drop_reasons'] = [reason for reason, is_dropped in zip(reasons, drop) if is_dropped]
    dropped.to_csv(os.path.join(args.output_dir, 'dropped.csv'), index=False)

    kept = merged[~drop][DATASET_COLUMNS].reset_index(drop=True)
    dataset = assign_patient_split(kept, seed=args.seed)
    dataset.to_csv(os.path.join(args.output_dir, 'dataset.csv'), index=False)
    print(f"Kept {len(dataset)} rows ({dataset['eyeid'].nunique()} eyes), dropped {len(dropped)} rows; "
          f"split counts {dataset['split'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
