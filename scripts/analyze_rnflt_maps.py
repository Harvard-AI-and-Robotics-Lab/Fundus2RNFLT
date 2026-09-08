"""Analyse the predicted RNFLT maps against the OCT maps.

Inputs: --csv = dataset_pred.csv (QC-filtered rows with rnflt_corr_path and pred_rnflt_path);
dropped.csv is read from the same directory to rebuild the unfiltered table; --data-dir/vf.csv
supplies the visual-field total-deviation points.
Outputs under --output-dir: analysis_table.csv, invalid_rate_by_severity.csv,
mae_by_{severity,thickness,subgroup}.csv, sectoral_rnflt.csv, sectoral_summary.csv,
structure_function_{global,sectors}.csv, cprnflt_auc.csv."""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from fundus2rnflt.metrics import compute_map_metrics
from fundus2rnflt.sectoral import (SECTOR_ORDER, TD_COLUMNS, build_annulus, compute_sector_mean_td,
                                   compute_sectoral_stats, disc_center, get_annulus_sectors, load_eye_flipped)
from fundus2rnflt.subgroups import (AGE_LABELS, GROUP_COLUMNS, MD_BINS, MD_LABELS, THICKNESS_BINS,
                                    THICKNESS_LABELS, VFI_LABELS, add_labels_and_bins)

BASE_COLUMNS = ['eyeid', 'fundus_path', 'rnflt_path', 'rnflt_mask_path', 'glaucoma_label',
                'age', 'race', 'male', 'ethnicity', 'vfi', 'md', 'psdprob']
QC_COLUMNS = ['invalid_tissue_pct', 'valid_pct', 'gt300_pct_valid']
SECTORAL_COLUMNS = ['eyeid', 'fundus_path', 'split', 'sector', 'n_pix', 'mean_um_corr', 'mean_um_pred', 'signed_error', 'mae']
SECTORAL_STATS = ['mean_um_corr', 'mean_um_pred', 'signed_error', 'mae']
ALL_SECTORS = SECTOR_ORDER + ['global']
SMALL_N_WARN = 25


def parse_args():
    p = argparse.ArgumentParser(description="Predicted-map analysis: MAE stratification, GH sectors, structure-function, cpRNFLT AUC")
    p.add_argument('--csv', type=str, required=True, help='dataset_pred.csv (dropped.csv is read from its directory)')
    p.add_argument('--data-dir', type=str, required=True, help='directory holding vf.csv')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--split', type=str, default='test', help='split used for the MAE tables, structure-function and AUC')
    p.add_argument('--radius', type=float, default=65, help='circumpapillary annulus radius (pixels)')
    p.add_argument('--band', type=float, default=3, help='annulus half-width (pixels)')
    return p.parse_args()


def build_analysis_table(dataset, dropped):
    """Unfiltered rows (kept + QC-dropped) with QC, corrected/predicted map paths, corrected-map
    thickness, per-fundus map metrics against the original OCT map, and severity."""
    table = pd.concat([dataset[BASE_COLUMNS + QC_COLUMNS], dropped[BASE_COLUMNS + QC_COLUMNS]], ignore_index=True)
    # corrected map and split are per eye; the predicted map is per (eye, fundus image)
    per_eye = dataset.drop_duplicates(subset='eyeid')[['eyeid', 'rnflt_corr_path', 'split']]
    table = table.merge(per_eye, on='eyeid', how='left')
    per_image = dataset.drop_duplicates(subset=['eyeid', 'fundus_path'])[['eyeid', 'fundus_path', 'pred_rnflt_path']]
    table = table.merge(per_image, on=['eyeid', 'fundus_path'], how='left')
    table['filtered'] = table['eyeid'].isin(set(dataset['eyeid']))

    eyes = table.dropna(subset=['rnflt_corr_path']).drop_duplicates(subset='eyeid')[['eyeid', 'rnflt_corr_path', 'rnflt_mask_path']]
    corr_records = []
    for _, row in tqdm(eyes.iterrows(), total=len(eyes), desc="Corrected-map thickness"):
        corr = np.load(row['rnflt_corr_path']).astype(np.float32)
        mask = np.load(row['rnflt_mask_path']).astype(np.uint8)
        corr_pixels = corr[(mask == 1) | (mask == 0)]  # everything except disc / cup
        corr_records.append({'eyeid': row['eyeid'],
                             'mean_um_corr': float(np.mean(corr_pixels)),
                             'std_um_corr': float(np.std(corr_pixels))})
    table = table.merge(pd.DataFrame(corr_records), on='eyeid', how='left')

    pairs = table.dropna(subset=['pred_rnflt_path']).drop_duplicates(subset=['eyeid', 'pred_rnflt_path'])
    metric_records = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Map metrics"):
        pred = np.load(row['pred_rnflt_path']).astype(np.float32)
        real = np.load(row['rnflt_path']).astype(np.float32)
        mask = np.load(row['rnflt_mask_path']).astype(np.uint8)
        metric_records.append({'eyeid': row['eyeid'], 'pred_rnflt_path': row['pred_rnflt_path'],
                               **compute_map_metrics(pred, real, mask)})
    table = table.merge(pd.DataFrame(metric_records), on=['eyeid', 'pred_rnflt_path'], how='left')

    table['severity'] = pd.cut(table['md'], bins=MD_BINS, labels=MD_LABELS)
    return table


def invalid_rate_by_severity(table):
    crosstab = pd.crosstab(table['severity'], table['filtered'], margins=True)
    crosstab = crosstab.reindex(index=MD_LABELS + ['All'], columns=[False, True, 'All'], fill_value=0)
    crosstab['filtered_rate'] = crosstab[False] / crosstab['All']
    result = crosstab[[False, 'All', 'filtered_rate']]
    result.columns = ['Invalid', 'Total', 'Invalid Rate']
    result.index.name = 'severity'
    return result


def error_by_group(df, group_col, order):
    """mean / std (ddof=1) / count of mae and signed_error per level of group_col, in the given order."""
    mae = df.groupby(group_col, observed=True)['mae'].agg(['mean', 'std', 'count']).reindex(order)
    signed = df.groupby(group_col, observed=True)['signed_error'].agg(['mean', 'std', 'count']).reindex(order)
    out = pd.DataFrame({'mae_mean': mae['mean'], 'mae_std': mae['std'], 'n': mae['count'],
                        'signed_error_mean': signed['mean'], 'signed_error_std': signed['std']})
    out.index.name = group_col
    return out


def subgroup_levels(df, col):
    """Level order of notebook 10: bin labels (Missing last) for the binned columns, alphabetical
    with Missing last otherwise."""
    if col == 'age_bin':
        fixed = AGE_LABELS
    elif col == 'md_bin':
        fixed = MD_LABELS
    elif col == 'vfi_bin':
        fixed = VFI_LABELS
    else:
        fixed = None
    present = [str(x) for x in df[col].unique()]
    if fixed is not None:
        return list(fixed) + (['Missing'] if 'Missing' in present else [])
    levels = sorted(present)
    if 'Missing' in levels:
        levels = [x for x in levels if x != 'Missing'] + ['Missing']
    return levels


def map_metric_summary(df, group, level):
    return {
        'group': group,
        'level': level,
        'n': int(len(df)),
        'mae_mean': float(df['mae'].mean()),
        'mae_std': float(df['mae'].std(ddof=1)),
        'ssim_mean': float(df['ssim'].mean()),
        'ssim_std': float(df['ssim'].std(ddof=1)),
        'valid_fraction_mean': float(df['valid_fraction'].mean()),
        'valid_fraction_std': float(df['valid_fraction'].std(ddof=1)),
    }


def mae_by_subgroup(split_rows):
    """Notebook 10: MAE / SSIM / valid fraction overall and per demographic or severity subgroup."""
    df = add_labels_and_bins(split_rows)
    df['valid_fraction'] = df['valid_pct'] / 100.0
    rows = [map_metric_summary(df, 'overall', 'all')]
    for col in GROUP_COLUMNS:
        for level in subgroup_levels(df, col):
            sub = df[df[col].astype(str) == level]
            if len(sub) == 0:
                continue
            rows.append(map_metric_summary(sub, col, level))
    out = pd.DataFrame(rows)
    out['small_n_warning'] = out['n'] < SMALL_N_WARN
    return out


def compute_sectoral_table(filtered, radius, band):
    """Garway-Heath sector statistics of the corrected vs predicted map for every filtered row,
    computed once per unique (eyeid, fundus image) and repeated for its duplicate rows."""
    blocks = {}
    rows = []
    for _, sample in tqdm(filtered.iterrows(), total=len(filtered), desc="Sectoral stats"):
        key = (sample['eyeid'], sample['fundus_path'])
        if key not in blocks:
            rnflt_corr, mask = load_eye_flipped(sample['rnflt_corr_path'], sample['rnflt_mask_path'], sample['eyeid'])
            rnflt_pred, _ = load_eye_flipped(sample['pred_rnflt_path'], sample['rnflt_mask_path'], sample['eyeid'])
            center = disc_center(mask)
            annulus = build_annulus(rnflt_corr.shape, center, radius=radius, band=band)
            labels = get_annulus_sectors(rnflt_corr.shape, center, annulus)
            blocks[key] = compute_sectoral_stats(rnflt_corr, rnflt_pred, annulus, labels)
        for stats in blocks[key]:
            rows.append({**stats, 'eyeid': sample['eyeid'], 'fundus_path': sample['fundus_path'], 'split': sample['split']})
    sectoral = pd.DataFrame(rows)[SECTORAL_COLUMNS]
    assert len(sectoral) == 7 * len(filtered)
    return sectoral


def sectoral_summary(sectoral, split):
    parts = []
    for subset, rows in (('all', sectoral), (split, sectoral[sectoral['split'] == split])):
        summary = rows.groupby('sector')[SECTORAL_STATS].agg(['mean', 'std']).reindex(ALL_SECTORS)
        summary.columns = [f"{stat}_{agg}" for stat, agg in summary.columns]
        summary.insert(0, 'n', len(rows) // 7)
        summary.insert(0, 'subset', subset)
        summary.index.name = 'sector'
        parts.append(summary.reset_index())
    return pd.concat(parts, ignore_index=True)


def structure_function_global(global_rows):
    """Spearman and Pearson correlation of the annulus mean thickness with MD."""
    rows = []
    for source, column in (('real', 'mean_um_corr'), ('pred', 'mean_um_pred')):
        rho, p = spearmanr(global_rows[column], global_rows['md'], nan_policy='omit')
        pair = global_rows[[column, 'md']].dropna()
        r, p_pearson = pearsonr(pair[column], pair['md'])
        rows.append({'source': source, 'n': int(len(global_rows)), 'spearman_rho': rho, 'spearman_p': p,
                     'pearson_r': r, 'pearson_p': p_pearson})
    return pd.DataFrame(rows)


def structure_function_sectors(split_filtered, sectoral_split, vf):
    """Spearman correlation per GH sector of the sector mean thickness with the sector mean total
    deviation of the VF points. VF visits are joined on (eyeid, md)."""
    vf = vf.copy()
    vf['eyeid'] = vf['id'].astype(str) + '-' + vf['righteye'].astype(str)
    vf = vf.drop_duplicates(subset=['eyeid', 'md'])
    base = split_filtered.merge(vf[['eyeid', 'md'] + TD_COLUMNS], on=['eyeid', 'md'], how='left')
    assert len(base) == len(split_filtered)
    assert base['td1'].notna().all(), "Some rows have no visual-field points (join on eyeid, md failed)"

    sector_td = compute_sector_mean_td(base)
    sectors = sectoral_split[sectoral_split['sector'] != 'global'].reset_index(drop=True)
    assert len(sector_td) == len(sectors)
    assert (sector_td['sector'].values == np.tile(SECTOR_ORDER, len(base))).all()
    assert (sectors['sector'].values == np.tile(SECTOR_ORDER, len(base))).all()
    assert (sector_td[['eyeid', 'fundus_path']].values == sectors[['eyeid', 'fundus_path']].values).all()
    for col in ['mean_um_corr', 'mean_um_pred', 'signed_error', 'mae', 'n_pix', 'split']:
        sector_td[col] = sectors[col].values

    rows = []
    for sector in SECTOR_ORDER:
        sub = sector_td[sector_td['sector'] == sector]
        for source, column in (('real', 'mean_um_corr'), ('pred', 'mean_um_pred')):
            rho, p = spearmanr(sub[column], sub['mean_td'], nan_policy='omit')
            rows.append({'sector': sector, 'source': source, 'n': int(len(sub)), 'rho': rho, 'p_value': p})
    return pd.DataFrame(rows)


def cprnflt_auc(table, sectoral, split):
    """AUC of the negated annulus mean thickness (thinner = more glaucomatous) for the glaucoma label."""
    cp = sectoral.loc[sectoral['sector'] == 'global', ['eyeid', 'fundus_path', 'mean_um_corr', 'mean_um_pred']]
    cp = cp.drop_duplicates(subset=['eyeid', 'fundus_path'])
    cp = cp.rename(columns={'mean_um_corr': 'cprnflt_corr', 'mean_um_pred': 'cprnflt_pred'})
    merged = table.merge(cp, on=['eyeid', 'fundus_path'], how='left')
    assert merged.loc[merged['filtered'], 'cprnflt_corr'].notna().all()
    test = merged[(merged['split'] == split) & (merged['filtered'] == True)]
    y = test['glaucoma_label']
    rows = []
    for source, column in (('real', 'cprnflt_corr'), ('pred', 'cprnflt_pred')):
        rows.append({'source': source, 'n': int(len(y)), 'prevalence': float(y.mean()),
                     'auc': float(roc_auc_score(y, -test[column]))})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)

    dataset = pd.read_csv(args.csv)
    dropped = pd.read_csv(os.path.join(os.path.dirname(args.csv), 'dropped.csv'))
    table = build_analysis_table(dataset, dropped)
    table.to_csv(out('analysis_table.csv'), index=False)
    print(f"Analysis table: {len(table)} rows, {int(table['filtered'].sum())} filtered")

    invalid_rate_by_severity(table).to_csv(out('invalid_rate_by_severity.csv'))

    split_rows = table[table['split'] == args.split].dropna(subset=['mae']).copy()
    print(f"{args.split} rows with map metrics: {len(split_rows)}")
    error_by_group(split_rows, 'severity', MD_LABELS).to_csv(out('mae_by_severity.csv'))

    split_rows['thickness_group'] = pd.cut(split_rows['mean_um_corr'], bins=THICKNESS_BINS, labels=THICKNESS_LABELS)
    by_thickness = error_by_group(split_rows, 'thickness_group', THICKNESS_LABELS)
    all_filtered = table[table['filtered'] == True].dropna(subset=['mae'])
    thickness_all = pd.cut(all_filtered['mean_um_corr'], bins=THICKNESS_BINS, labels=THICKNESS_LABELS)
    counts_all = thickness_all.value_counts().reindex(THICKNESS_LABELS)
    by_thickness['n_all_splits'] = counts_all.values
    by_thickness['fraction_all_splits'] = (counts_all / counts_all.sum()).values
    by_thickness.to_csv(out('mae_by_thickness.csv'))

    mae_by_subgroup(split_rows).to_csv(out('mae_by_subgroup.csv'), index=False)

    filtered = table[table['filtered'] == True].reset_index(drop=True)
    sectoral = compute_sectoral_table(filtered, args.radius, args.band)
    sectoral.to_csv(out('sectoral_rnflt.csv'), index=False)
    sectoral_summary(sectoral, args.split).to_csv(out('sectoral_summary.csv'), index=False)

    # sectoral rows follow the filtered rows one block of 7 at a time, so md can be attached by position
    global_rows = sectoral[sectoral['sector'] == 'global'].reset_index(drop=True)
    assert (global_rows[['eyeid', 'fundus_path']].values == filtered[['eyeid', 'fundus_path']].values).all()
    global_rows['md'] = filtered['md'].values
    structure_function_global(global_rows[global_rows['split'] == args.split]).to_csv(out('structure_function_global.csv'), index=False)

    vf = pd.read_csv(os.path.join(args.data_dir, 'vf.csv'))
    split_filtered = filtered[filtered['split'] == args.split].reset_index(drop=True)
    sectoral_split = sectoral[sectoral['split'] == args.split].reset_index(drop=True)
    structure_function_sectors(split_filtered, sectoral_split, vf).to_csv(out('structure_function_sectors.csv'), index=False)

    cprnflt_auc(table, sectoral, args.split).to_csv(out('cprnflt_auc.csv'), index=False)
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == '__main__':
    main()
