"""OCT segmentation CSVs -> RNFL thickness map + mask, per-eye QC, drop rule, patient split."""
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PIXEL_SPACING = 1.95503  # µm per pixel along depth (ILM -> RNFL)
NATIVE_SHAPE = (200, 200)
MAP_SHAPE = (224, 224)

ILM_FILENAME = 'segmentation_ilm.csv'
RNFL_FILENAME = 'segmentation_rnfl_to_gcl.csv'
DISC_FILENAME = 'mask_disc.csv'
CUP_FILENAME = 'mask_cup.csv'


# ---------------------------------------------------------------------------
# Map + mask building
# ---------------------------------------------------------------------------
def has_segmentation(oct_dir):
    return os.path.exists(os.path.join(oct_dir, ILM_FILENAME)) and os.path.exists(os.path.join(oct_dir, RNFL_FILENAME))


def _load_seg(path, shape):
    arr = pd.read_csv(path, header=None).values.astype(np.float32).squeeze()
    assert arr.ndim == 1, f"Unexpected ndim for {path}: {arr.shape}"
    H, W = shape
    assert H * W == arr.size, f"Shape {shape} incompatible with n={arr.size} in {path}."
    return arr.reshape(H, W)


def _safe_read_mask_csv(path, shape):
    """Load a 0/1 mask if present; else zeros (eyes without a disc/cup segmentation)."""
    if os.path.exists(path):
        m = _load_seg(path, shape).astype(np.uint8)
        return (m == 1).astype(np.uint8)
    H, W = shape
    return np.zeros((H, W), dtype=np.uint8)


def _resize_thickness_with_mask(th_um, valid_bool, out_hw):
    """Validity-weighted bilinear resize: resize(thickness*valid) / resize(valid), NaN where denom ~ 0."""
    out_h, out_w = out_hw
    v_float = valid_bool.astype(np.float32)
    num = cv2.resize(th_um * v_float, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(v_float, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((out_h, out_w), np.nan, dtype=np.float32)
    m = den > 1e-6
    out[m] = num[m] / den[m]
    return out


def build_rnflt_map_and_mask(oct_dir):
    """
    Returns:
      rnflt_um : float32 (224,224), NaN where invalid or disc/cup
      mask_codes : uint8 (224,224): 0 = invalid/missing (not disc/cup), 1 = valid tissue, 2 = disc, 3 = cup
    """
    assert has_segmentation(oct_dir), f"Missing ILM/RNFL segmentation in {oct_dir}"
    ilm = _load_seg(os.path.join(oct_dir, ILM_FILENAME), NATIVE_SHAPE).astype(np.float32)
    rnfl = _load_seg(os.path.join(oct_dir, RNFL_FILENAME), NATIVE_SHAPE).astype(np.float32)

    # valid if finite & positive & ordered (RNFL > ILM)
    finite = np.isfinite(ilm) & np.isfinite(rnfl)
    positive = (ilm > 0) & (rnfl > 0)
    ordered = rnfl > ilm
    valid_native = finite & positive & ordered

    disc_native = _safe_read_mask_csv(os.path.join(oct_dir, DISC_FILENAME), NATIVE_SHAPE)
    cup_native = _safe_read_mask_csv(os.path.join(oct_dir, CUP_FILENAME), NATIVE_SHAPE)
    # cup overrides disc if both are 1
    disc_native = (disc_native == 1) & (cup_native == 0)
    cup_native = (cup_native == 1)

    valid_tissue_native = valid_native & (~disc_native) & (~cup_native)

    rnflt_um_native = (rnfl - ilm) * PIXEL_SPACING
    rnflt_um_native[~valid_tissue_native] = np.nan

    rnflt_um_resized = _resize_thickness_with_mask(rnflt_um_native, valid_tissue_native, MAP_SHAPE)

    out_h, out_w = MAP_SHAPE
    disc_res = cv2.resize(disc_native.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    cup_res = cv2.resize(cup_native.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    valid_frac_res = cv2.resize(valid_tissue_native.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    valid_res = (valid_frac_res >= 0.5).astype(np.uint8)

    mask_codes = np.zeros((out_h, out_w), dtype=np.uint8)
    mask_codes[valid_res == 1] = 1
    mask_codes[disc_res == 1] = 2
    mask_codes[cup_res == 1] = 3
    rnflt_um_resized[mask_codes != 1] = np.nan

    return rnflt_um_resized.astype(np.float32), mask_codes.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-eye QC
# ---------------------------------------------------------------------------
def _invalid_tissue_stats(mask_codes):
    """Invalid tissue coverage over the whole map (mask code 0, i.e. excluding disc/cup)."""
    total_px = mask_codes.size
    invalid_px = int((mask_codes == 0).sum())
    return {
        'invalid_tissue_px': invalid_px,
        'invalid_tissue_pct': invalid_px / total_px * 100.0,
    }


def _valid_stats(rnflt_um, mask_codes):
    """Thickness statistics on valid tissue only (mask code 1)."""
    valid = (mask_codes == 1)
    v = rnflt_um[np.isfinite(rnflt_um) & valid]
    if v.size == 0:
        return {
            'valid_px': 0, 'valid_pct': 0.0,
            'mean_um': np.nan, 'std_um': np.nan, 'p95_um': np.nan, 'p99_um': np.nan,
            'max_um': np.nan,
            'gt300_pct_valid': np.nan, 'gt350_pct_valid': np.nan,
        }
    total = rnflt_um.size
    return {
        'valid_px': int(v.size),
        'valid_pct': v.size / total * 100.0,
        'mean_um': float(np.mean(v)),
        'std_um': float(np.std(v)),
        'p95_um': float(np.percentile(v, 95)),
        'p99_um': float(np.percentile(v, 99)),
        'max_um': float(np.max(v)),
        'gt300_pct_valid': float((v > 300).mean() * 100.0),
        'gt350_pct_valid': float((v > 350).mean() * 100.0),
    }


def _disc_cup_coverage(mask_codes):
    disc = (mask_codes == 2).sum()
    cup = (mask_codes == 3).sum()
    total = mask_codes.size
    return {
        'disc_pct': disc / total * 100.0,
        'cup_pct': cup / total * 100.0,
        'masked_total_pct': (disc + cup) / total * 100.0,
    }


def compute_qc_for_eye(rnflt_path, mask_path):
    rnflt = np.load(rnflt_path)
    mask = np.load(mask_path).astype(np.uint8)
    H, W = mask.shape
    valid_stats = _valid_stats(rnflt, mask)
    return {
        'height': H, 'width': W, 'total_px': H * W,
        **_disc_cup_coverage(mask),
        **_invalid_tissue_stats(mask),
        **valid_stats,
        'no_valid_tissue': (valid_stats['valid_px'] == 0),
    }


def drop_mask(df, thresh_invalid=20.0, thresh_extreme=2.0):
    """Boolean drop flag and reason string per row of a table carrying the QC columns.
    Drop if QC is missing, there is no valid tissue, invalid tissue > thresh_invalid %,
    or more than thresh_extreme % of valid pixels exceed 300 µm."""
    cond_missing_qc = df[['invalid_tissue_pct', 'gt300_pct_valid']].isna().any(axis=1)
    cond_no_valid = (df['no_valid_tissue'] == True) | (df['valid_pct'].fillna(0) <= 0)
    cond_too_invalid = df['invalid_tissue_pct'] > thresh_invalid
    cond_too_extreme = df['gt300_pct_valid'] > thresh_extreme
    drop = cond_missing_qc | cond_no_valid | cond_too_invalid | cond_too_extreme

    reasons = []
    for i in range(len(df)):
        r = []
        if cond_missing_qc.iloc[i]:
            r.append('qc_missing')
        if cond_no_valid.iloc[i]:
            r.append('no_valid_tissue')
        if cond_too_invalid.iloc[i]:
            r.append(f'invalid>{thresh_invalid:.0f}%')
        if cond_too_extreme.iloc[i]:
            r.append(f'>300um>{thresh_extreme:.0f}%_of_valid')
        reasons.append(';'.join(r) if r else 'keep')
    return drop, reasons


# ---------------------------------------------------------------------------
# Patient-level stratified split
# ---------------------------------------------------------------------------
def _patient_label(series):
    """Mode of the patient's non-NaN glaucoma labels."""
    s = series.dropna()
    assert len(s) > 0, "Patient with no glaucoma label"
    return s.mode().iloc[0]


def _stratify_labels(ids, label_map):
    ys = np.array([label_map[pid] for pid in ids], dtype=int)
    _, counts = np.unique(ys, return_counts=True)
    assert np.all(counts >= 2), "Stratified split needs at least two patients per class"
    return ys


def assign_patient_split(df, seed=42):
    """Add patient_id (eyeid prefix) and split (train 80 / val 10 / test 10 by patient,
    stratified on the patient-level glaucoma label)."""
    df = df.copy()
    df['patient_id'] = df['eyeid'].str.split('-').str[0]
    unique_patients = df['patient_id'].unique()

    patient_labels = df.groupby('patient_id')['glaucoma_label'].apply(_patient_label).astype(int)
    label_map = patient_labels.to_dict()

    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.2,
        stratify=_stratify_labels(unique_patients, label_map), random_state=seed)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5,
        stratify=_stratify_labels(temp_patients, label_map), random_state=seed)

    def assign_split(row):
        pid = row['patient_id']
        if pid in train_patients:
            return 'train'
        if pid in val_patients:
            return 'val'
        return 'test'

    df['split'] = df.apply(assign_split, axis=1)

    splits_per_patient = df.groupby('patient_id')['split'].nunique()
    assert (splits_per_patient == 1).all(), "A patient appears in multiple splits"
    return df
