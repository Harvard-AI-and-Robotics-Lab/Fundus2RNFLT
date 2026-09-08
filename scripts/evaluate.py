"""Evaluate trained glaucoma classifiers.

Per run (<runs-dir>/<exp>/<run>/ holding model.pth, config.json and, for RNFLT runs,
rnflt_norm_stats.json): re-run inference on the val and test splits, choose the decision threshold
on val (Youden's J), and write predictions, the threshold, overall and stratified test metrics.
Across runs: leaderboard.csv, and for every pair of runs pairwise_delong.csv (AUROC overall and per
demographic stratum) and pairwise_bootstrap_severity.csv (accuracy per MD bin, paired bootstrap)."""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fundus2rnflt.datasets import EvalDataset
from fundus2rnflt.metrics import (acc_metric_factory, bootstrap_metric_diff, brier_score, delong_roc_test,
                                  expected_calibration_error, safe_auprc, safe_auroc,
                                  summarize_overall_and_strata, youdens_j)
from fundus2rnflt.models import build_classifier_from_config
from fundus2rnflt.subgroups import add_labels_and_bins

PREDICTION_COLUMNS = ["eyeid", "y_true", "y_prob", "age", "male", "race", "ethnicity", "md", "vfi"]
DEMOGRAPHIC_COLUMNS = ["sex_label", "race_label", "ethnicity_label", "age_bin", "vfi_bin"]
SMALL_N_STRATA = 25
SMALL_N_BOOTSTRAP = 10


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate glaucoma classifier runs")
    p.add_argument('--csv', type=str, required=True, help='dataset CSV with split, labels, demographics and map paths')
    p.add_argument('--runs-dir', type=str, required=True,
                   help='directory holding <exp>/<run>/model.pth (+ config.json, rnflt_norm_stats.json)')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--runs', type=str, nargs='*', default=None,
                   help='run keys <exp>/<run> to evaluate (default: every run under --runs-dir)')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


def discover_runs(runs_dir):
    run_keys = []
    for exp_name in sorted(os.listdir(runs_dir)):
        exp_dir = os.path.join(runs_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        for run_id in sorted(os.listdir(exp_dir)):
            if os.path.exists(os.path.join(exp_dir, run_id, 'model.pth')):
                run_keys.append(f"{exp_name}/{run_id}")
    assert len(run_keys) > 0, f"No <exp>/<run>/model.pth found under {runs_dir}"
    return run_keys


def rnflt_source_for(input_type):
    if input_type == 'fundus':
        return 'none'
    if input_type in ('rnflt_real', 'fused_real'):
        return 'real'
    if input_type in ('rnflt_pred', 'fused_pred'):
        return 'predicted'
    raise ValueError(input_type)


def load_rnflt_stats(config, run_dir):
    """(mean, std) for z-score normalisation: from the config if given, else from the stats the
    training script saved next to the checkpoint."""
    if config['rnflt_mean'] is not None and config['rnflt_std'] is not None:
        return float(config['rnflt_mean']), float(config['rnflt_std'])
    with open(os.path.join(run_dir, 'rnflt_norm_stats.json')) as fp:
        stats = json.load(fp)
    assert stats['mode'] == 'zscore'
    assert stats['std'] > 0
    return float(stats['mean']), float(stats['std'])


@torch.no_grad()
def infer_split(model, dual_branch, loader, device):
    model.eval()
    columns = {key: [] for key in PREDICTION_COLUMNS}
    for batch in loader:
        if dual_branch:
            logits = model(batch['fundus'].to(device), batch['rnflt'].to(device))
        else:
            parts = []
            if 'fundus' in batch:
                parts.append(batch['fundus'])
            if 'rnflt' in batch:
                parts.append(batch['rnflt'])
            x = torch.cat(parts, dim=1).to(device) if len(parts) > 1 else parts[0].to(device)
            logits = model(x)

        prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        columns['y_true'].extend(batch['label'].numpy().astype(int).tolist())
        columns['y_prob'].extend(prob.tolist())
        columns['eyeid'].extend(batch['eyeid'])
        for key in ('age', 'male', 'race', 'ethnicity', 'md', 'vfi'):
            columns[key].extend(batch[key].numpy().tolist())
    return pd.DataFrame(columns)


def evaluate_run(run_key, run_dir, csv_path, results_dir, device, batch_size, num_workers):
    """Inference on val/test, threshold on val, metrics on test. Returns the test predictions,
    the threshold and the leaderboard row."""
    with open(os.path.join(run_dir, 'config.json')) as fp:
        config = json.load(fp)
    input_type = config['input_type']
    use_fundus = input_type in ('fundus', 'fused_real', 'fused_pred')
    rnflt_source = rnflt_source_for(input_type)
    rnflt_norm = config['rnflt_norm']
    rnflt_stats = None
    if rnflt_source != 'none' and rnflt_norm == 'zscore':
        rnflt_stats = load_rnflt_stats(config, run_dir)

    model, dual_branch = build_classifier_from_config(config)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pth'), map_location=device, weights_only=True))
    model.eval()

    def make_loader(split):
        dataset = EvalDataset(
            csv_path=csv_path, split=split, use_fundus=use_fundus,
            rnflt_source=rnflt_source, rnflt_channels=int(config['rnflt_channels']),
            rnflt_norm=rnflt_norm, rnflt_stats=rnflt_stats,
            rnflt_minmax=(float(config['rnflt_min']), float(config['rnflt_max'])),
            rnflt_fill=float(config['rnflt_fill']))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    val_df = infer_split(model, dual_branch, make_loader('val'), device)
    test_df = infer_split(model, dual_branch, make_loader('test'), device)

    os.makedirs(os.path.join(results_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    val_df.to_parquet(os.path.join(results_dir, 'predictions', 'val_predictions.parquet'), index=False)
    test_df.to_parquet(os.path.join(results_dir, 'predictions', 'test_predictions.parquet'), index=False)

    val_df_b = add_labels_and_bins(val_df)
    test_df_b = add_labels_and_bins(test_df)

    yv, pv = val_df_b['y_true'].values, val_df_b['y_prob'].values
    thr, thr_info = youdens_j(yv, pv)
    with open(os.path.join(results_dir, 'threshold.json'), 'w') as fp:
        json.dump({"threshold": float(thr), **thr_info}, fp, indent=2)

    overall_df, strat_df = summarize_overall_and_strata(test_df_b, thr, small_n_warn=SMALL_N_STRATA)
    overall_df.to_csv(os.path.join(results_dir, 'metrics', 'overall_test_metrics.csv'), index=False)
    strat_df.to_csv(os.path.join(results_dir, 'metrics', 'stratified_test_metrics.csv'), index=False)

    val_overall = {
        "n": int(len(val_df_b)),
        "auroc": safe_auroc(yv, pv),
        "auprc": safe_auprc(yv, pv),
        "brier": brier_score(yv, pv),
        "ece_10": expected_calibration_error(yv, pv, 10),
    }
    pd.DataFrame([val_overall]).to_csv(os.path.join(results_dir, 'metrics', 'val_threshold_free.csv'), index=False)

    overall = overall_df.iloc[0]
    run_id = run_key.split('/')[1]
    leaderboard_row = {
        "label": f"{input_type}/{config['model_name']} [{run_id}]",
        "input_type": input_type,
        "model_name": config['model_name'],
        "n_test": int(overall["n"]),
        "auroc": overall["auroc"],
        "acc": overall["thr@accuracy"],
        "bal_acc": overall["thr@balanced_accuracy"],
        "threshold": float(thr),
    }
    print(f"{run_key}: test n={leaderboard_row['n_test']} auroc={leaderboard_row['auroc']:.4f} "
          f"acc={leaderboard_row['acc']:.4f} threshold={thr:.4f}")
    return test_df_b, thr, leaderboard_row


def align_two_runs(test_a, test_b):
    """Test predictions of two runs share the row order of the dataset CSV; assert it and return one
    frame with y_prob_A / y_prob_B."""
    assert len(test_a) == len(test_b), "Runs have different test-set sizes"
    assert np.array_equal(test_a['y_true'].values, test_b['y_true'].values), "y_true mismatch between runs"
    assert np.array_equal(test_a['eyeid'].values, test_b['eyeid'].values), "eyeid order mismatch between runs"
    df = test_a.rename(columns={'y_prob': 'y_prob_A'})
    df['y_prob_B'] = test_b['y_prob'].values
    return df


def delong_rows(run_a, run_b, df):
    """AUROC of both runs with DeLong z / p overall and per demographic stratum. Strata with fewer
    than two samples of either class are skipped."""
    rows = []
    strata = [('overall', 'all', df)]
    for group_col in DEMOGRAPHIC_COLUMNS:
        for level in sorted(df[group_col].unique(), key=str):
            strata.append((group_col, str(level), df[df[group_col] == level]))
    for group, level, sub in strata:
        y = sub['y_true'].values
        if (y == 1).sum() < 2 or (y == 0).sum() < 2:
            continue
        auc_a, auc_b, z, p_value = delong_roc_test(y, sub['y_prob_A'].values, sub['y_prob_B'].values)
        rows.append({"run_a": run_a, "run_b": run_b, "group": group, "level": level, "n": int(len(sub)),
                     "auc_a": auc_a, "auc_b": auc_b, "z": z, "p_value": p_value})
    return rows


def bootstrap_severity_rows(run_a, run_b, label_a, label_b, df, thr_a, thr_b, n_boot, seed):
    """Paired bootstrap of accuracy (each run at its own threshold) per MD severity bin."""
    acc_a = acc_metric_factory(thr_a)
    acc_b = acc_metric_factory(thr_b)
    rows = []
    for level in sorted(df['md_bin'].dropna().unique(), key=str):
        sub = df[df['md_bin'] == level]
        n = len(sub)
        if n == 0:
            continue
        result = bootstrap_metric_diff(sub['y_true'].values, sub['y_prob_A'].values, sub['y_prob_B'].values,
                                       metric_fn_A=acc_a, metric_fn_B=acc_b, n_boot=n_boot, seed=seed)
        rows.append({
            "run_a": run_a, "run_b": run_b,
            "group": "md_bin", "level": str(level), "n": int(n), "metric_type": "acc",
            "modelA_label": label_a, "modelB_label": label_b,
            "metric_A_full": result["metric_A_full"],
            "metric_B_full": result["metric_B_full"],
            "metric_A_ci_low": result["metric_A_ci"][0],
            "metric_A_ci_high": result["metric_A_ci"][1],
            "metric_B_ci_low": result["metric_B_ci"][0],
            "metric_B_ci_high": result["metric_B_ci"][1],
            "delta_mean": result["delta_mean"],
            "delta_ci_low": result["delta_ci"][0],
            "delta_ci_high": result["delta_ci"][1],
            "t_stat": result["t_stat"],
            "p_value": result["p_value"],
            "n_boot_valid": result["n_boot_valid"],
            "small_n_warning": (n < SMALL_N_BOOTSTRAP),
        })
    return rows


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_keys = args.runs if args.runs else discover_runs(args.runs_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    test_predictions = {}
    thresholds = {}
    leaderboard_rows = []
    for run_key in run_keys:
        run_dir = os.path.join(args.runs_dir, run_key)
        results_dir = os.path.join(args.output_dir, run_key)
        test_df_b, thr, leaderboard_row = evaluate_run(run_key, run_dir, args.csv, results_dir, device,
                                                       args.batch_size, args.num_workers)
        test_predictions[run_key] = test_df_b
        thresholds[run_key] = thr
        leaderboard_rows.append(leaderboard_row)

    leaderboard = (pd.DataFrame(leaderboard_rows)
                   .sort_values(["auroc", "bal_acc", "acc"], ascending=False)
                   .reset_index(drop=True))
    leaderboard.to_csv(os.path.join(args.output_dir, 'leaderboard.csv'), index=False)
    print(leaderboard.to_string(index=False))

    if len(run_keys) < 2:
        print("Only one run evaluated: no pairwise comparisons")
        return

    model_labels = {run_key: f"{row['input_type']}/{row['model_name']}"
                    for run_key, row in zip(run_keys, leaderboard_rows)}
    delong = []
    bootstrap = []
    for run_a, run_b in itertools.combinations(run_keys, 2):
        df = align_two_runs(test_predictions[run_a], test_predictions[run_b])
        delong.extend(delong_rows(run_a, run_b, df))
        bootstrap.extend(bootstrap_severity_rows(run_a, run_b, model_labels[run_a], model_labels[run_b], df,
                                                 thresholds[run_a], thresholds[run_b], args.n_boot, args.seed))
    pd.DataFrame(delong).to_csv(os.path.join(args.output_dir, 'pairwise_delong.csv'), index=False)
    (pd.DataFrame(bootstrap).sort_values(["run_a", "run_b", "group", "level"]).reset_index(drop=True)
     .to_csv(os.path.join(args.output_dir, 'pairwise_bootstrap_severity.csv'), index=False))
    print(f"Wrote leaderboard and {len(delong)} DeLong / {len(bootstrap)} bootstrap rows for "
          f"{len(run_keys) * (len(run_keys) - 1) // 2} run pairs to {args.output_dir}")


if __name__ == '__main__':
    main()
