"""Demographic and severity subgroup definitions shared by evaluation and map analysis."""
import numpy as np
import pandas as pd

RACE_MAP = {7: "White", 3: "Black", 2: "Asian", 1: "Other", 4: "Other", 5: "Other", 6: "Other"}
ETHN_MAP = {0: "Non-Hispanic", 1: "Hispanic"}

AGE_BINS = [-np.inf, 49, 59, 69, 79, np.inf]
AGE_LABELS = ["<50", "50-59", "60-69", "70-79", "≥80"]

MD_BINS = [-np.inf, -12, -6, -3, np.inf]
MD_LABELS = ["≤-12 (Severe)", "-12 to -6 (Moderate)", "-6 to -3 (Mild)", ">-3 (None)"]

VFI_BINS = [-np.inf, 69, 89, 98, np.inf]
VFI_LABELS = ["<70", "70–89", "90–98", "≥99"]

THICKNESS_BINS = [-np.inf, 60, 70, 80, 90, np.inf]
THICKNESS_LABELS = ["<60 µm", "60-70 µm", "70-80 µm", "80-90 µm", ">90 µm"]

GROUP_COLUMNS = ["sex_label", "race_label", "ethnicity_label", "age_bin", "md_bin", "vfi_bin"]


def sex_label(male_value):
    if pd.isna(male_value):
        return "Missing"
    return "Male" if int(male_value) == 1 else "Female"


def _bin_with_missing(values, bins, labels):
    binned = pd.cut(values, bins=bins, labels=labels, right=True)
    return binned.astype(object).where(binned.notna(), "Missing")


def add_labels_and_bins(df):
    """Add race_label, ethnicity_label, sex_label, age_bin, md_bin, vfi_bin columns."""
    out = df.copy()
    out["race_label"] = out["race"].map(RACE_MAP).fillna("Missing")
    out["ethnicity_label"] = out["ethnicity"].map(ETHN_MAP).fillna("Missing")
    out["sex_label"] = out["male"].apply(sex_label)
    out["age_bin"] = _bin_with_missing(out["age"], AGE_BINS, AGE_LABELS)
    out["md_bin"] = _bin_with_missing(out["md"], MD_BINS, MD_LABELS)
    out["vfi_bin"] = _bin_with_missing(out["vfi"], VFI_BINS, VFI_LABELS)
    return out
