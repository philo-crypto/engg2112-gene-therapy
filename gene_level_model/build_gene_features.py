#!/usr/bin/env python3
"""
Build gene-level features for the static genomic risk model.

Label: mean log1p(clonal_abundance at last observed timepoint) across all
       patients who had an integration site near the gene.

Features: 15 pure genomic columns only (no temporal/observational data).
          Gene-level features are identical for every site near the same gene
          so we just take the first value after joining.

Output: gene_features.csv (~15k rows, one per unique nearest_gene_name)
"""

import os
import re
import numpy as np
import pandas as pd

ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VIS_PATH     = os.path.join(ROOT, "data", "vis_raw.csv")
TRAIN_PATH   = os.path.join(ROOT, "training_data.csv")
OUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gene_features.csv")

GENOMIC_COLS = [
    "dist_to_nearest_tss",
    "in_gene_body",
    "dist_to_nearest_cancer_tss",
    "gene_length",
    "in_repeat",
    "cpg_island_count",
    "cpg_nearest_dist",
    "cpg_mean_obs_exp",
    "encode_ccre_count",
    "encode_promoter_count",
    "encode_enhancer_count",
    "encode_ctcf_count",
    "ot_cancer_hallmark_count",
    "ot_disease_count",
    "ot_is_cancer_driver",
]


def parse_timepoint_weeks(tp) -> float:
    if tp is None or (isinstance(tp, float) and np.isnan(tp)):
        return np.nan
    s = str(tp).strip().lower()
    m = re.search(r"(\d+\.?\d*)\s*m", s)
    if m:
        return float(m.group(1)) * 4.33
    m = re.search(r"(\d+\.?\d*)\s*w", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*y", s)
    if m:
        return float(m.group(1)) * 52
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return np.nan


# -- 1. Last-timepoint abundance per patient-site -----------------------------
print("Loading vis_raw.csv...")
vis = pd.read_csv(VIS_PATH, low_memory=False)
print(f"  {len(vis):,} rows")

vis["tp_weeks"] = vis["timepoint"].apply(parse_timepoint_weeks)

# For sites where timepoint is unknown, fall back to max abundance row
vis["tp_weeks_filled"] = vis["tp_weeks"].fillna(-1)

last_idx = (
    vis.groupby(["patient_id", "chrom", "site_pos"])["tp_weeks_filled"]
    .idxmax()
)
last = vis.loc[last_idx].copy()
last["label"] = np.log1p(last["clonal_abundance"])

print(f"  {len(last):,} unique patient-site pairs after taking last timepoint")
print(f"  Label range: {last['label'].min():.3f} - {last['label'].max():.3f}")

# -- 2. Load genomic features from training_data.csv --------------------------
print("\nLoading training_data.csv for genomic features...")
train = pd.read_csv(TRAIN_PATH, low_memory=False, usecols=[
    "chrom", "site_pos", "nearest_gene_name", "nearest_cancer_gene_name",
] + GENOMIC_COLS)
print(f"  {len(train):,} rows, {train['nearest_gene_name'].nunique():,} unique genes")

# One genomic feature row per unique site (drop patient duplicates)
site_features = train.drop_duplicates(subset=["chrom", "site_pos"])

# -- 3. Join last-timepoint labels to genomic features ------------------------
print("\nJoining...")
merged = last.merge(
    site_features[["chrom", "site_pos", "nearest_gene_name"] + GENOMIC_COLS],
    on=["chrom", "site_pos"],
    how="inner",
)
print(f"  {len(merged):,} patient-site pairs with genomic features")
print(f"  {merged['nearest_gene_name'].nunique():,} unique genes covered")

# -- 4. Aggregate to gene level -----------------------------------------------
print("\nAggregating to gene level...")
agg = (
    merged.groupby("nearest_gene_name")
    .agg(
        chrom              = ("chrom",        "first"),
        label              = ("label",         "mean"),
        label_max          = ("label",         "max"),
        n_unique_patients  = ("patient_id",    "nunique"),
        n_site_obs         = ("site_pos",      "count"),
        **{col: (col, "first") for col in GENOMIC_COLS},
    )
    .reset_index()
)

print(f"  -> {len(agg):,} genes")
print(f"  Label range: {agg['label'].min():.3f} - {agg['label'].max():.3f}")
print(f"  Label mean:  {agg['label'].mean():.3f}  std: {agg['label'].std():.3f}")

agg.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> gene_level_model/gene_features.csv")
print(f"Columns: {agg.columns.tolist()}")
