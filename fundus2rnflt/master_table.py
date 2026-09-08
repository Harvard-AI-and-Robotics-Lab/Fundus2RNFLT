"""Raw hospital tables (fundus, OCT, VF) -> aligned master table with the VF-based glaucoma label."""
import numpy as np
import pandas as pd

FUNDUS_OCT_MAX_DAYS = 180
VF_OCT_MAX_DAYS = 30

KEEP_CORE_COLUMNS = [
    # identity & date
    'id', 'righteye', 'timeoftest_fundus', 'timeoftest_oct', 'timeoftest',
    # paths
    'jpgfile', 'datadir',
    # demographics
    'age', 'race', 'hispanic', 'male',
    # OCT quality / geometry
    'signalstrength', 'bscanx', 'bscany', 'bscanPixelspacingDepth',
    # VF reliability
    'malfixnum', 'malfixdenom', 'falsenegrate', 'falseposrate',
    # glaucoma label
    'glaucoma_label', 'md', 'ght', 'psdprob', 'vfi', 'psd',
    # OCT summaries
    'avgthickness', 'discarea', 'cupvol', 'rimarea', 'verticalcdratio', 'avgcdratio',
    # offsets
    'scanpatternoffsetx', 'scanpatternoffsety',
]


def _days_between(a, b):
    return (pd.to_datetime(a) - pd.to_datetime(b)).abs().dt.days


def build_master_table(fundus, oct_, vf):
    """Keep eyes present in all three tables, align the tables row by row on patient-eye,
    and merge them. Fundus columns get the suffix _fundus, OCT columns _oct, VF columns none."""
    fundus = fundus.copy()
    oct_ = oct_.copy()
    vf = vf.copy()
    for table in (fundus, oct_, vf):
        table['pt_eye'] = table['id'].astype(str) + '_' + table['righteye'].astype(str)

    # Stable sort: visits of the same eye keep the raw tables' (chronological) order.
    common_eyes = set(fundus['pt_eye']) & set(oct_['pt_eye']) & set(vf['pt_eye'])
    assert len(common_eyes) > 0, "No patient-eye is present in all three tables"
    fundus = fundus[fundus['pt_eye'].isin(common_eyes)].sort_values('pt_eye', kind='stable').reset_index(drop=True)
    oct_ = oct_[oct_['pt_eye'].isin(common_eyes)].sort_values('pt_eye', kind='stable').reset_index(drop=True)
    vf = vf[vf['pt_eye'].isin(common_eyes)].sort_values('pt_eye', kind='stable').reset_index(drop=True)

    assert len(fundus) == len(oct_) == len(vf), "Dataset sizes don't match"
    assert (fundus['pt_eye'].values == oct_['pt_eye'].values).all(), "Fundus-OCT alignment failed"
    assert (fundus['pt_eye'].values == vf['pt_eye'].values).all(), "Fundus-VF alignment failed"

    fundus['match_index'] = fundus.index
    oct_['match_index'] = oct_.index
    vf['match_index'] = vf.index
    master = (fundus
              .merge(oct_, on='match_index', suffixes=('_fundus', '_oct'))
              .merge(vf, on='match_index'))

    assert (master['id_fundus'] == master['id_oct']).all()
    assert (master['id_fundus'] == master['id']).all()
    assert (master['righteye_fundus'] == master['righteye_oct']).all()

    fundus_oct_days = _days_between(master['timeoftest_fundus'], master['timeoftest_oct'])
    vf_oct_days = _days_between(master['timeoftest'], master['timeoftest_oct'])
    assert (fundus_oct_days <= FUNDUS_OCT_MAX_DAYS).all(), "Fundus photo more than 180 days from OCT"
    assert (vf_oct_days <= VF_OCT_MAX_DAYS).all(), "Visual field more than 30 days from OCT"
    return master


def assign_glaucoma_label(master):
    """glaucoma_label: 1 if MD < -3 and GHT abnormal (3) and PSD prob > 1;
    0 if MD >= -1 and GHT normal (1) and PSD prob <= 1; NaN otherwise."""
    master = master.copy()
    ght_abnormal = master['ght'] == 3
    ght_normal = master['ght'] == 1
    psd_abnormal = master['psdprob'] > 1
    psd_normal = master['psdprob'] <= 1
    md_abnormal = master['md'] < -3.0
    md_normal = master['md'] >= -1.0

    master['glaucoma_label'] = np.nan
    master.loc[md_abnormal & ght_abnormal & psd_abnormal, 'glaucoma_label'] = 1
    master.loc[md_normal & ght_normal & psd_normal, 'glaucoma_label'] = 0
    return master


def select_core_columns(master):
    return master[KEEP_CORE_COLUMNS].copy()
