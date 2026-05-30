#!/usr/bin/env python3
"""
Build patient-gene feature table.

For each patient, identifies which genes had confirmed integrations at the
FIRST observed timepoint, then labels them with abundance at the LAST
observed timepoint (0 if the clone was no longer detectable).

Rows: (patient_id, nearest_gene_name) pairs with confirmed time-1 integration.
Non-integrated genes are excluded entirely — not assumed safe.

Label: log1p(max clonal abundance at last timepoint for sites near that gene,
             for that patient; 0 if undetectable at last timepoint)
"""

import os
import re
import numpy as np
import pandas as pd

ROOT       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VIS_PATH   = os.path.join(ROOT, "data", "vis_raw.csv")
TRAIN_PATH = os.path.join(ROOT, "training_data.csv")
OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "patient_gene_features.csv")

GENOMIC_COLS = [
    "dist_to_nearest_tss",
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


# -- 1. Load VIS data ---------------------------------------------------------
print("Loading vis_raw.csv...")
vis = pd.read_csv(VIS_PATH, low_memory=False)
vis["tp_weeks"] = vis["timepoint"].apply(parse_timepoint_weeks)

print(f"  {len(vis):,} observations, {vis['patient_id'].nunique()} patients")
print(f"  Timepoints with parseable weeks: {vis['tp_weeks'].notna().sum():,} / {len(vis):,}")

# Per-patient first and last timepoint
# For patients with unknown timepoints (single measurement), treat as both first and last
vis["patient_min_tp"] = vis.groupby("patient_id")["tp_weeks"].transform("min")
vis["patient_max_tp"] = vis.groupby("patient_id")["tp_weeks"].transform("max")

# Patients with all-NaN timepoints: assign a sentinel so they appear in both first/last
sentinel = -1.0
vis.loc[vis["patient_min_tp"].isna(), "patient_min_tp"] = sentinel
vis.loc[vis["patient_max_tp"].isna(), "patient_max_tp"] = sentinel
vis["tp_weeks"] = vis["tp_weeks"].fillna(sentinel)

print("\nPer-patient timepoint range:")
tp_summary = (
    vis.groupby("patient_id")
    .agg(first_tp=("tp_weeks", "min"), last_tp=("tp_weeks", "max"),
         n_timepoints=("tp_weeks", "nunique"))
    .round(1)
)
print(tp_summary.to_string())

# -- 2. First-timepoint integration events ------------------------------------
# Sites observed at the patient's earliest timepoint = confirmed integrations
first_obs = vis[vis["tp_weeks"] == vis["patient_min_tp"]].copy()
first_sites = first_obs[["patient_id", "chrom", "site_pos"]].drop_duplicates()
print(f"\nFirst-timepoint integration sites: {len(first_sites):,} unique patient-site pairs")
print(f"  Per-patient counts:")
for pid, n in first_sites.groupby("patient_id").size().items():
    print(f"    {pid}: {n} sites")

# -- 3. Last-timepoint abundances ---------------------------------------------
last_obs = vis[vis["tp_weeks"] == vis["patient_max_tp"]].copy()
last_agg = (
    last_obs
    .groupby(["patient_id", "chrom", "site_pos"])["clonal_abundance"]
    .max()
    .reset_index()
    .rename(columns={"clonal_abundance": "last_tp_abundance"})
)
print(f"\nLast-timepoint observations: {len(last_agg):,} unique patient-site pairs")

# -- 4. Load gene mappings from training_data.csv -----------------------------
print("\nLoading gene mappings from training_data.csv...")
usecols = ["chrom", "site_pos", "nearest_gene_name"] + GENOMIC_COLS
train = pd.read_csv(TRAIN_PATH, low_memory=False, usecols=usecols)

# One genomic feature row per unique site
site_gene = train.drop_duplicates(subset=["chrom", "site_pos"])
print(f"  {len(site_gene):,} unique sites with gene mappings")

# -- 5. Map first-tp sites to genes -------------------------------------------
first_mapped = first_sites.merge(
    site_gene[["chrom", "site_pos", "nearest_gene_name"]],
    on=["chrom", "site_pos"], how="inner"
)
print(f"\nFirst-tp sites mapped to genes: {len(first_mapped):,} / {len(first_sites):,}")

# Unique (patient, gene) pairs with confirmed time-1 integration
patient_gene = first_mapped[["patient_id", "nearest_gene_name"]].drop_duplicates()
print(f"Unique (patient, gene) pairs: {len(patient_gene):,}")

# Add chrom back (take first gene occurrence — gene is on one chromosome)
gene_chrom = (
    site_gene[["nearest_gene_name", "chrom"]]
    .drop_duplicates(subset="nearest_gene_name", keep="first")
)
patient_gene = patient_gene.merge(gene_chrom, on="nearest_gene_name", how="left")

# -- 6. Map last-tp sites to genes and get abundance --------------------------
last_mapped = last_agg.merge(
    site_gene[["chrom", "site_pos", "nearest_gene_name"]],
    on=["chrom", "site_pos"], how="inner"
)
last_gene_agg = (
    last_mapped
    .groupby(["patient_id", "nearest_gene_name"])["last_tp_abundance"]
    .max()
    .reset_index()
)

# -- 7. Build label -----------------------------------------------------------
# Join: for every confirmed (patient, gene), get last-tp abundance
# Genes where the clone disappeared get abundance = 0
df = patient_gene.merge(last_gene_agg, on=["patient_id", "nearest_gene_name"], how="left")
df["last_tp_abundance"] = df["last_tp_abundance"].fillna(0)
df["label"] = np.log1p(df["last_tp_abundance"])

print(f"\nLabel distribution:")
print(f"  Rows with label = 0 (clone not detectable at last tp): "
      f"{(df['label'] == 0).sum():,} / {len(df):,} "
      f"({(df['label']==0).mean()*100:.1f}%)")
print(f"  Label range: {df['label'].min():.3f} - {df['label'].max():.3f}")
print(f"  Label mean:  {df['label'].mean():.3f}  std: {df['label'].std():.3f}")

# -- 8. Add genomic features --------------------------------------------------
gene_features = (
    site_gene.drop_duplicates(subset="nearest_gene_name", keep="first")
    [["nearest_gene_name"] + GENOMIC_COLS]
)
df = df.merge(gene_features, on="nearest_gene_name", how="left")

# Label-encode patient
df["patient_id_enc"] = pd.Categorical(df["patient_id"]).codes

# -- 9. Save ------------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} rows x {len(df.columns)} cols -> patient_gene_features.csv")
print(f"Columns: {df.columns.tolist()}")
print(f"Chromosomes covered: {sorted(df['chrom'].unique())}")
