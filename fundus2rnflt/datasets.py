"""Datasets for the fundus -> RNFLT U-Net, the glaucoma classifiers, and classifier evaluation."""
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rnflt_and_mask(row, rnflt_key, mask_key):
    """Load RNFLT (float32 HxW with NaNs) and a boolean mask of valid tissue.
    If the row has no usable mask file, every finite pixel counts as valid."""
    rnflt_path = row[rnflt_key]
    assert os.path.exists(rnflt_path), f"Missing RNFLT file: {rnflt_path}"
    rnflt = np.load(rnflt_path).astype(np.float32)

    if mask_key and (mask_key in row) and isinstance(row[mask_key], str) and os.path.exists(row[mask_key]):
        mask_codes = np.load(row[mask_key]).astype(np.uint8)  # 0=invalid,1=valid,2=disc,3=cup
        valid_mask = (mask_codes == 1)
    else:
        valid_mask = np.isfinite(rnflt)

    valid_mask = valid_mask & np.isfinite(rnflt)
    return rnflt, valid_mask


def _to_tensor_img(img_pil, transform_fundus):
    if transform_fundus is not None:
        return transform_fundus(img_pil)
    return T.ToTensor()(img_pil)


def _to_tensor_map(arr_hw):
    """HxW -> 1xHxW float32 torch tensor."""
    return torch.from_numpy(arr_hw.astype(np.float32)).unsqueeze(0)


def _to_tensor_mask(mask_bool_hw):
    """Boolean HxW -> 1xHxW float32 {0,1} tensor."""
    return torch.from_numpy(mask_bool_hw.astype(np.float32)).unsqueeze(0)


def _load_rnflt_filled(path, fill_value):
    arr = np.load(path).astype(np.float32)
    return np.where(np.isfinite(arr), arr, fill_value)


def compute_rnflt_stats(train_df, rnflt_key, fill_value=0.0):
    """Mean/std over NaN-filled RNFLT maps of the train split (for z-score normalisation)."""
    maps = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing RNFLT stats"):
        maps.append(_load_rnflt_filled(row[rnflt_key], fill_value))
    stacked = np.stack(maps, axis=0)
    mean = float(stacked.mean())
    std = float(stacked.std() + 1e-8)
    return mean, std


# ---------------------------------------------------------------------------
# Fundus -> RNFLT (Mode A and Mode B2)
# ---------------------------------------------------------------------------
class FundusToRNFLDataset(Dataset):
    """
    Mode A (use_corrected=False):
      target = original rnflt_path (NaNs filled with rnflt_fill_value)
      rnflt_mask = valid tissue only: (mask_codes == 1) & isfinite(original)
      returns {'fundus', 'rnflt', 'rnflt_mask'}

    Mode B2 (use_corrected=True):
      target = corrected rnflt_corr_path (NaNs filled with rnflt_fill_value)
      rnflt_mask = valid tissue only (same as Mode A, used for validation)
      weight map: 1.0 on valid tissue (mask_codes==1); corr_weight on newly supervised pixels
      (originally invalid, or originally thin: original < thin_threshold within valid tissue);
      0.0 on disc/cup (mask_codes in {2,3}). Returned as the additional key 'weight'.
    """
    def __init__(self, csv_path, transform_fundus=None, transform_rnflt=None,
                 rnflt_fill_value=0.0, use_corrected=False, corr_weight=0.3, thin_threshold=20.0):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.tf = transform_fundus
        self.tr = transform_rnflt
        self.rnflt_key = 'rnflt_path'
        self.rnflt_mask_key = 'rnflt_mask_path'
        self.corrected_key = 'rnflt_corr_path'
        self.rnflt_fill_value = rnflt_fill_value
        self.use_corrected = bool(use_corrected)
        self.corr_weight = float(corr_weight)
        self.thin_threshold = float(thin_threshold)

        if self.use_corrected:
            assert self.corrected_key in self.df.columns, f"CSV must contain '{self.corrected_key}' when use_corrected=True."
            has_corrected = (self.df[self.corrected_key].notna()
                             & self.df[self.corrected_key].map(lambda p: isinstance(p, str) and os.path.exists(p)))
            self.df = self.df[has_corrected].reset_index(drop=True)

        assert self.rnflt_mask_key in self.df.columns, f"CSV must contain '{self.rnflt_mask_key}'."

    def __len__(self):
        return len(self.df)

    def _load_original_and_mask(self, row):
        """Original RNFLT (HxW float32, may contain NaNs), mask codes (HxW uint8), valid tissue (bool)."""
        rnflt_path = row[self.rnflt_key]
        assert isinstance(rnflt_path, str) and os.path.exists(rnflt_path), f"Missing RNFLT file: {rnflt_path}"
        rnflt_orig = np.load(rnflt_path).astype(np.float32)

        mask_path = row[self.rnflt_mask_key]
        assert isinstance(mask_path, str) and os.path.exists(mask_path), f"Missing RNFLT mask file: {mask_path}"
        mask_codes = np.load(mask_path).astype(np.uint8)

        valid_tissue = (mask_codes == 1) & np.isfinite(rnflt_orig)
        return rnflt_orig, mask_codes, valid_tissue

    def _load_target(self, row):
        if self.use_corrected:
            target_path = row[self.corrected_key]
        else:
            target_path = row[self.rnflt_key]
        return _load_rnflt_filled(target_path, self.rnflt_fill_value)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fundus_img = Image.open(row['fundus_path']).convert('RGB')
        fundus = _to_tensor_img(fundus_img, self.tf)

        rnflt_orig, mask_codes, valid_tissue = self._load_original_and_mask(row)

        rnflt = _to_tensor_map(self._load_target(row))
        if self.tr is not None:
            rnflt = self.tr(rnflt)

        sample = {
            'fundus': fundus,                          # 3xHxW
            'rnflt': rnflt,                            # 1xHxW (target)
            'rnflt_mask': _to_tensor_mask(valid_tissue),  # 1xHxW (valid tissue only; used in val)
        }

        if self.use_corrected:
            H, W = mask_codes.shape
            weight = np.zeros((H, W), dtype=np.float32)
            disc_cup = (mask_codes == 2) | (mask_codes == 3)
            valid = (mask_codes == 1)
            weight[valid] = 1.0
            originally_invalid = (mask_codes == 0)
            originally_thin = valid & np.isfinite(rnflt_orig) & (rnflt_orig < self.thin_threshold)
            weight[originally_invalid | originally_thin] = self.corr_weight
            weight[disc_cup] = 0.0
            sample['weight'] = _to_tensor_map(weight)

        return sample


# ---------------------------------------------------------------------------
# Glaucoma classification (fundus and/or RNFLT) for training
# ---------------------------------------------------------------------------
class GlaucomaDataset(Dataset):
    """
    Returns dict with keys depending on configuration:
      - fundus: 3xHxW
      - rnflt:  CxHxW where C=1 (RNFLT only) or C=2 (RNFLT + binary mask) if include_rnflt_mask_channel
      - rnflt_mask: 1xHxW
      - label:  float tensor (0/1)
    rnflt_source: 'real' (rnflt_path + rnflt_mask_path) | 'predicted' (pred_rnflt_path) | 'none'
    """
    def __init__(self, csv_path, transform_fundus=None, transform_rnflt=None, use_fundus=True,
                 rnflt_source='real', include_rnflt_mask_channel=False, rnflt_fill_value=0.0):
        assert rnflt_source in ('real', 'predicted', 'none')
        self.df = pd.read_csv(csv_path)
        self.tf = transform_fundus
        self.tr = transform_rnflt
        self.use_fundus = use_fundus
        self.rnflt_source = rnflt_source
        self.include_rnflt_mask_channel = include_rnflt_mask_channel
        self.rnflt_fill_value = rnflt_fill_value

        if rnflt_source == 'real':
            self.rnflt_key = 'rnflt_path'
            self.mask_key = 'rnflt_mask_path'
        elif rnflt_source == 'predicted':
            self.rnflt_key = 'pred_rnflt_path'
            self.mask_key = 'pred_rnflt_mask_path'  # optional; absent -> mask derived from finite values
        else:
            self.rnflt_key = None
            self.mask_key = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        if self.use_fundus:
            fundus_img = Image.open(row['fundus_path']).convert('RGB')
            sample['fundus'] = _to_tensor_img(fundus_img, self.tf)

        if self.rnflt_source != 'none':
            rnflt_np, valid_mask_np = _load_rnflt_and_mask(row, self.rnflt_key, self.mask_key)
            rnflt_np_filled = np.where(np.isfinite(rnflt_np), rnflt_np, self.rnflt_fill_value)

            rnflt = _to_tensor_map(rnflt_np_filled)
            rnflt_mask = _to_tensor_mask(valid_mask_np)
            if self.tr is not None:
                rnflt = self.tr(rnflt)

            sample['rnflt'] = rnflt
            sample['rnflt_mask'] = rnflt_mask
            if self.include_rnflt_mask_channel:
                sample['rnflt'] = torch.cat([rnflt, rnflt_mask], dim=0)  # 2xHxW

        sample['label'] = torch.tensor(int(row['glaucoma_label']), dtype=torch.float32)
        return sample


# ---------------------------------------------------------------------------
# Glaucoma classification: evaluation-only dataset with metadata
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """
    Eval-only dataset over one split. Produces:
      - fundus tensor [3,H,W] if use_fundus
      - rnflt tensor [C,H,W] (C = rnflt_channels: 1 = map, 2 = [map, mask]) if rnflt_source != 'none'
      - label, eyeid, age, male, race, ethnicity, md, vfi
    """
    def __init__(self, csv_path, split, use_fundus, rnflt_source='none', rnflt_channels=1,
                 rnflt_norm='zscore', rnflt_stats=None, rnflt_minmax=(0.0, 350.0), rnflt_fill=0.0):
        df = pd.read_csv(csv_path)
        assert split in ('train', 'val', 'test')
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.use_fundus = use_fundus
        self.rnflt_source = rnflt_source
        self.rnflt_channels = rnflt_channels
        self.rnflt_norm = rnflt_norm
        self.rnflt_stats = rnflt_stats
        self.rnflt_minmax = rnflt_minmax
        self.rnflt_fill = rnflt_fill

        self.tf_fundus = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _rnflt_key(self):
        if self.rnflt_source == 'real':
            return 'rnflt_path'
        if self.rnflt_source == 'predicted':
            assert 'pred_rnflt_path' in self.df.columns, "CSV missing 'pred_rnflt_path' required for rnflt_source='predicted'."
            return 'pred_rnflt_path'
        raise ValueError(self.rnflt_source)

    def _norm_rnflt(self, arr):
        x = torch.from_numpy(arr).float()
        if self.rnflt_norm == 'zscore':
            assert self.rnflt_stats is not None and self.rnflt_stats[1] > 0, "zscore requires rnflt mean/std."
            mean, std = self.rnflt_stats
            x = (x - mean) / std
        elif self.rnflt_norm == 'minmax':
            vmin, vmax = self.rnflt_minmax
            denom = max(vmax - vmin, 1e-6)
            x = (x - vmin) / denom
        elif self.rnflt_norm == 'none':
            pass
        else:
            raise ValueError(self.rnflt_norm)
        return x.unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        out = {}

        if self.use_fundus:
            img = Image.open(row['fundus_path']).convert('RGB')
            out['fundus'] = self.tf_fundus(img)

        if self.rnflt_source != 'none':
            arr = _load_rnflt_filled(row[self._rnflt_key()], self.rnflt_fill)
            rnflt_tensor = self._norm_rnflt(arr)
            if self.rnflt_channels == 2:
                if isinstance(row.get('rnflt_mask_path', None), str) and os.path.exists(row['rnflt_mask_path']):
                    codes = np.load(row['rnflt_mask_path']).astype(np.uint8)
                    mask = (codes == 1)
                else:
                    mask = np.isfinite(arr)
                mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
                rnflt_tensor = torch.cat([rnflt_tensor, mask_t], dim=0)
            out['rnflt'] = rnflt_tensor

        out['label'] = torch.tensor(int(row['glaucoma_label']), dtype=torch.long)
        out['eyeid'] = row['eyeid']
        out['age'] = row['age']
        out['male'] = row['male']
        out['race'] = row['race']
        out['ethnicity'] = row['ethnicity']
        out['md'] = row['md']
        out['vfi'] = row['vfi']
        return out
