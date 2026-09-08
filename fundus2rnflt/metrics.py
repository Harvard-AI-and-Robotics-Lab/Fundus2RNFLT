"""Losses and metrics: masked map losses, map-quality metrics, classification metrics,
paired bootstrap, and DeLong tests."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from fundus2rnflt.subgroups import GROUP_COLUMNS


# ---------------------------------------------------------------------------
# Masked losses for the fundus -> RNFLT U-Net (torch)
# ---------------------------------------------------------------------------
def masked_mae(pred, target, mask):
    """pred/target: Bx1xHxW. mask: Bx1xHxW in {0,1} (1 = include in loss)."""
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred, target, mask):
        diff = pred - target
        abs_diff = torch.abs(diff)
        delta = torch.tensor(self.delta, device=diff.device, dtype=diff.dtype)
        quadratic = torch.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom


# ---------------------------------------------------------------------------
# Map-quality metrics on numpy arrays
# ---------------------------------------------------------------------------
def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)


def masked_ssim_numpy(pred_hw, real_hw, valid_mask, prefer_win_size=7):
    """SSIM restricted to the bounding box of valid pixels. Each map is min-max normalised
    over its valid pixels, invalid pixels are filled with 0.5, and the window is the largest
    odd size <= prefer_win_size that fits."""
    assert valid_mask.sum() > 0, "masked_ssim_numpy needs at least one valid pixel"
    y0, y1, x0, x1 = _bbox_from_mask(valid_mask)
    pc = pred_hw[y0:y1, x0:x1].astype(np.float32, copy=True)
    rc = real_hw[y0:y1, x0:x1].astype(np.float32, copy=True)
    vm = valid_mask[y0:y1, x0:x1].astype(bool)

    pv = pc[vm]
    rv = rc[vm]
    pmin, pmax = float(pv.min()), float(pv.max())
    rmin, rmax = float(rv.min()), float(rv.max())
    prange = max(pmax - pmin, 1e-6)
    rrange = max(rmax - rmin, 1e-6)

    p_norm = (pc - pmin) / prange
    r_norm = (rc - rmin) / rrange

    fill_value = 0.5
    inv = ~vm
    if inv.any():
        p_norm[inv] = fill_value
        r_norm[inv] = fill_value

    h, w = p_norm.shape
    win_size = min(prefer_win_size, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    return float(structural_similarity(r_norm, p_norm, data_range=1.0, win_size=win_size))


def compute_map_metrics(pred, real, mask_codes):
    """MAE, signed error (pred - real) and masked SSIM over valid tissue (mask code 1, finite real)."""
    valid_mask = (mask_codes == 1) & np.isfinite(real)
    diff = pred[valid_mask] - real[valid_mask]
    return {
        "mae": float(np.abs(diff).mean()),
        "signed_error": float(diff.mean()),
        "ssim": masked_ssim_numpy(pred, real, valid_mask),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def youdens_j(y_true, y_prob):
    """Threshold maximising TPR - FPR on the ROC curve. Returns (threshold, info dict)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    assert np.isfinite(thr[idx]), "No ROC point beats chance; Youden's J gives no finite threshold"
    return float(thr[idx]), {
        "criterion": "youdens_j",
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "index": int(idx),
    }


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with equal-width bins in [0,1]."""
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(in_bin):
            continue
        conf = y_prob[in_bin].mean()
        acc = (y_true[in_bin] >= 0.5).mean()
        ece += in_bin.mean() * abs(acc - conf)
    return float(ece)


def thresholded_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(y_prob) >= thr).astype(int)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    rec = sens
    f1 = (2 * prec * rec) / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
    balacc = (sens + spec) / 2.0
    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(balacc),
    }


def safe_auroc(y, p):
    """AUROC, or NaN when only one class is present."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    """Average precision, or NaN when only one class is present."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def summarize_overall_and_strata(df_pred, thr, small_n_warn=25):
    """df_pred: columns y_true, y_prob and the GROUP_COLUMNS from subgroups.add_labels_and_bins.
    Returns (overall_metrics_df, stratified_metrics_df)."""
    y = df_pred["y_true"].values
    p = df_pred["y_prob"].values
    overall = {}
    overall["n"] = int(len(df_pred))
    overall["auroc"] = safe_auroc(y, p)
    overall["auprc"] = safe_auprc(y, p)
    overall["brier"] = brier_score(y, p)
    overall["ece_10"] = expected_calibration_error(y, p, n_bins=10)
    overall.update({f"thr@{k}": v for k, v in thresholded_metrics(y, p, thr).items()})
    overall_df = pd.DataFrame([overall])

    rows = []
    for col in GROUP_COLUMNS:
        for lvl in sorted(df_pred[col].unique(), key=lambda x: str(x)):
            sub = df_pred[df_pred[col] == lvl]
            if len(sub) == 0:
                continue
            mets = thresholded_metrics(sub["y_true"].values, sub["y_prob"].values, thr)
            row = {
                "group": col,
                "level": str(lvl),
                "n": mets["n"],
                "auroc": safe_auroc(sub["y_true"].values, sub["y_prob"].values),
                "auprc": safe_auprc(sub["y_true"].values, sub["y_prob"].values),
                "small_n_warning": mets["n"] < small_n_warn,
                **{k: v for k, v in mets.items() if k != "n"},
            }
            rows.append(row)
    strat_df = pd.DataFrame(rows).sort_values(["group", "level"]).reset_index(drop=True)
    return overall_df, strat_df


# ---------------------------------------------------------------------------
# Paired bootstrap comparison of two models
# ---------------------------------------------------------------------------
def auc_metric(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def acc_metric_factory(threshold):
    """Return a metric function that computes accuracy at a fixed threshold."""
    def acc_metric(y, p):
        y = np.asarray(y).astype(int)
        p = np.asarray(p).astype(float)
        if len(y) == 0:
            return np.nan
        y_hat = (p >= threshold).astype(int)
        return float((y_hat == y).mean())
    return acc_metric


def bootstrap_metric_diff(y, pA, pB, metric_fn_A, metric_fn_B, n_boot=1000, seed=12345):
    """Paired bootstrap of metric_B - metric_A over resampled rows. Returns full-sample
    metrics, 95% percentile CIs, the mean delta with its CI, and a one-sample t-test
    of the bootstrap deltas against zero. Resamples where either metric is NaN are skipped."""
    y = np.asarray(y)
    pA = np.asarray(pA)
    pB = np.asarray(pB)
    n = len(y)
    rng = np.random.default_rng(seed)

    metric_A_full = metric_fn_A(y, pA)
    metric_B_full = metric_fn_B(y, pB)

    metrics_A = []
    metrics_B = []
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mA = metric_fn_A(y[idx], pA[idx])
        mB = metric_fn_B(y[idx], pB[idx])
        if np.isnan(mA) or np.isnan(mB):
            continue
        metrics_A.append(mA)
        metrics_B.append(mB)
        deltas.append(mB - mA)

    result = {
        "metric_A_full": float(metric_A_full),
        "metric_B_full": float(metric_B_full),
        "n_boot_valid": int(len(deltas)),
    }

    if len(deltas) == 0:
        result.update({
            "metric_A_ci": (np.nan, np.nan),
            "metric_B_ci": (np.nan, np.nan),
            "delta_mean": np.nan,
            "delta_ci": (np.nan, np.nan),
            "t_stat": np.nan,
            "p_value": np.nan,
        })
        return result

    metrics_A = np.asarray(metrics_A)
    metrics_B = np.asarray(metrics_B)
    deltas = np.asarray(deltas)

    result["metric_A_ci"] = (float(np.percentile(metrics_A, 2.5)), float(np.percentile(metrics_A, 97.5)))
    result["metric_B_ci"] = (float(np.percentile(metrics_B, 2.5)), float(np.percentile(metrics_B, 97.5)))
    result["delta_mean"] = float(deltas.mean())
    result["delta_ci"] = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

    if len(deltas) >= 2 and np.std(deltas, ddof=1) > 0:
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_val)
    else:
        result["t_stat"] = np.nan
        result["p_value"] = np.nan

    return result


# ---------------------------------------------------------------------------
# DeLong test for correlated ROC curves (Sun & Xu 2014 midrank algorithm)
# ---------------------------------------------------------------------------
def _midrank(values):
    """Average rank (1-based) of each element, ties share the mean rank."""
    return stats.rankdata(values, method='average')


def fast_delong(y_true, scores):
    """y_true: (n,) binary labels. scores: (k, n) scores from k classifiers.
    Returns (aucs (k,), delong_covariance (k, k))."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    assert np.isfinite(scores).all(), "DeLong needs finite scores"
    positive = y_true == 1
    num_positive = int(positive.sum())
    num_negative = int((~positive).sum())
    assert num_positive > 1 and num_negative > 1, "DeLong needs at least two samples of each class"

    positive_scores = scores[:, positive]
    negative_scores = scores[:, ~positive]
    num_classifiers = scores.shape[0]
    positive_ranks = np.empty((num_classifiers, num_positive))
    negative_ranks = np.empty((num_classifiers, num_negative))
    overall_ranks = np.empty((num_classifiers, num_positive + num_negative))
    for r in range(num_classifiers):
        positive_ranks[r] = _midrank(positive_scores[r])
        negative_ranks[r] = _midrank(negative_scores[r])
        overall_ranks[r] = _midrank(np.concatenate([positive_scores[r], negative_scores[r]]))
    aucs = overall_ranks[:, :num_positive].sum(axis=1) / num_positive / num_negative - (num_positive + 1.0) / (2.0 * num_negative)
    # Structural components: per positive, the fraction of negatives it outranks; per negative, the fraction of positives above it.
    positive_components = (overall_ranks[:, :num_positive] - positive_ranks) / num_negative
    negative_components = 1.0 - (overall_ranks[:, num_positive:] - negative_ranks) / num_positive
    delong_covariance = np.cov(positive_components) / num_positive + np.cov(negative_components) / num_negative
    return aucs, np.atleast_2d(delong_covariance)


def delong_roc_test(y_true, scores_a, scores_b):
    """Two-sided DeLong test that two correlated AUCs differ.
    Returns (auc_a, auc_b, z, p_value). Identical scores give z = 0, p = 1."""
    aucs, cov = fast_delong(y_true, np.stack([scores_a, scores_b]))
    variance = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if variance == 0.0:
        assert aucs[0] == aucs[1], "Zero DeLong variance with different AUCs (degenerate scores); no test possible"
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p_value)


def delong_auc_ci(y_true, scores, alpha=0.05):
    """AUC with its (1 - alpha) DeLong confidence interval, clipped to [0, 1]."""
    aucs, cov = fast_delong(y_true, np.asarray(scores)[None, :])
    auc = float(aucs[0])
    standard_error = float(np.sqrt(cov[0, 0]))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = max(0.0, auc - z_crit * standard_error)
    upper = min(1.0, auc + z_crit * standard_error)
    return auc, lower, upper
