"""Circumpapillary annulus sampling, Garway-Heath sectors, and the VF point -> sector lookup."""
import numpy as np
import pandas as pd

ANNULUS_RADIUS = 65
ANNULUS_BAND = 3

# (name, abbreviation, start angle, end angle). 0° = temporal, 90° = superior, 180° = nasal, 270° = inferior.
GH_SECTORS = [
    ("Temporal", "T", 315.0, 45.0),
    ("Superotemporal", "ST", 45.0, 85.0),
    ("Superonasal", "SN", 85.0, 125.0),
    ("Nasal", "N", 125.0, 235.0),
    ("Inferonasal", "IN", 235.0, 275.0),
    ("Inferotemporal", "IT", 275.0, 315.0),
]
SECTOR_ORDER = ["T", "ST", "SN", "N", "IN", "IT"]


def load_eye_flipped(rnflt_path, mask_path, eyeid):
    """Load a map and its mask, flipping left eyes (eyeid ending in 0) to OD orientation."""
    rnflt = np.load(rnflt_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.uint8)
    if eyeid.split('-')[-1] == '0':
        rnflt = np.fliplr(rnflt)
        mask = np.fliplr(mask)
    return rnflt, mask


def disc_center(mask):
    """(cy, cx) centroid of disc + cup pixels."""
    disc = (mask == 2) | (mask == 3)
    assert disc.any(), "No disc/cup pixels in mask"
    ys, xs = np.where(disc)
    return float(ys.mean()), float(xs.mean())


def build_annulus(shape, center, radius=ANNULUS_RADIUS, band=ANNULUS_BAND):
    """Pixels with (radius - band) <= distance_to_center <= (radius + band)."""
    H, W = shape
    cy, cx = center
    yy, xx = np.indices((H, W))
    rr = np.hypot(yy - cy, xx - cx)
    return (rr >= (radius - band)) & (rr <= (radius + band))


def get_pixel_angles(shape, center):
    """Polar angle of each pixel about the center, in degrees. Assumes OD orientation."""
    H, W = shape
    cy, cx = center
    yy, xx = np.indices((H, W))
    return np.degrees(np.arctan2(cy - yy, cx - xx)) % 360.0


def angle_to_sector(angles):
    """Sector index 0..5 (order of GH_SECTORS) for each angle."""
    angles = np.asarray(angles) % 360.0
    out = np.full(angles.shape, -1, dtype=np.int8)
    for i, (name, abbr, lo, hi) in enumerate(GH_SECTORS):
        if lo < hi:
            sector_mask = (angles >= lo) & (angles < hi)
        else:  # wrap-around for temporal
            sector_mask = (angles >= lo) | (angles < hi)
        out[sector_mask] = i
    return out


def get_annulus_sectors(shape, center, annulus_mask):
    """Sector label per pixel inside the annulus; -1 outside."""
    sectors = angle_to_sector(get_pixel_angles(shape, center)).copy()
    sectors[~annulus_mask] = -1
    return sectors


def compute_sectoral_stats(rnflt_corr, rnflt_pred, annulus, labels):
    """Per-sector (six GH sectors + 'global' annulus) mean thickness, MAE and signed error
    over pixels where both maps are finite. Returns a list of dicts."""
    sectors = [(abbr, labels == i) for i, (_, abbr, _, _) in enumerate(GH_SECTORS)]
    sectors.append(('global', annulus))

    rows = []
    diff = rnflt_pred - rnflt_corr
    for name, sector in sectors:
        sector_corr = rnflt_corr[sector]
        sector_pred = rnflt_pred[sector]
        sector_diff = diff[sector]
        finite = np.isfinite(sector_corr) & np.isfinite(sector_pred)
        sector_corr, sector_pred, sector_diff = sector_corr[finite], sector_pred[finite], sector_diff[finite]
        rows.append({
            'sector': name,
            'n_pix': int(sector_corr.size),
            'mean_um_corr': float(sector_corr.mean()),
            'mean_um_pred': float(sector_pred.mean()),
            'mae': float(np.abs(sector_diff).mean()),
            'signed_error': float(sector_diff.mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Visual field: 24-2 point -> Garway-Heath sector
# ---------------------------------------------------------------------------
HFA_INDEX = list(range(1, 55))
SECTOR_RAW = [
    "IN", "IN", "IN", "IN", "IN", "IT", "IT", "IT", "IN", "IN",
    "IT", "IT", "IT", "IT", "IT", "IT", "IN", "N", "IT", "IT",
    "IT", "IT", "T", "T", "T", "bs", "N", "SN", "ST", "ST",
    "ST", "T", "T", "T", "bs", "N", "SN", "ST", "ST", "ST",
    "ST", "ST", "SN", "N", "SN", "SN", "ST", "ST", "SN", "SN",
    "SN", "SN", "SN", "SN",
]
assert len(HFA_INDEX) == len(SECTOR_RAW) == 54

vf_point_gh_lookup = pd.DataFrame({
    'hfa_index': HFA_INDEX,
    'is_blind_spot': [s == 'bs' for s in SECTOR_RAW],
    'sector': [None if s == 'bs' else s for s in SECTOR_RAW],
})
assert vf_point_gh_lookup["is_blind_spot"].sum() == 2

BLIND_SPOT_POINTS = (26, 35)
TD_COLUMNS = [f"td{i}" for i in range(1, 55) if i not in BLIND_SPOT_POINTS]

HFA_INDEX_TO_SECTOR = (
    vf_point_gh_lookup.loc[~vf_point_gh_lookup["is_blind_spot"], ["hfa_index", "sector"]]
    .set_index("hfa_index")["sector"]
)


def compute_sector_mean_td(df):
    """Mean total deviation of the VF points in each GH sector. One row per sample x sector
    (rows ordered by input row, then SECTOR_ORDER)."""
    rows = []
    for _, row in df.iterrows():
        buckets = {s: [] for s in SECTOR_ORDER}
        for col in TD_COLUMNS:
            point_index = int(col[2:])  # td23 -> 23
            buckets[HFA_INDEX_TO_SECTOR[point_index]].append(row[col])
        for s in SECTOR_ORDER:
            rows.append({
                "eyeid": row["eyeid"],
                "fundus_path": row["fundus_path"],
                "md": row["md"],
                "sector": s,
                "mean_td": np.mean(buckets[s]),
            })
    return pd.DataFrame(rows)
