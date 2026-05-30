#!/usr/bin/env python3
"""
Patient-gene feature table v2 — adds growth_rate per (patient, gene) pair.

Growth rate = linear slope of log1p(clonal_abundance) over time (per week),
fitted across all observed timepoints for sites near that gene for that patient.

Row    = (patient_id, nearest_gene_name) with confirmed integration at time 1.
Label  = log1p(abundance at last timepoint; 0 if clone undetectable).
Features = 15 genomic cols + growth_rate + n_timepoints_observed.

Note: for patients with only 2 timepoints, growth_rate correlates with the
label by construction. n_timepoints_observed flags this in the data.
"""

import os
import re
import numpy as np
import pandas as pd

ROOT       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VIS_PATH   = os.path.join(ROOT, "data", "vis_raw.csv")
TRAIN_PATH = os.path.join(ROOT, "training_data.csv")
OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "patient_gene_features_v3.csv")

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


# -- 1. Load VIS --------------------------------------------------------------
print("Loading vis_raw.csv...")
vis = pd.read_csv(VIS_PATH, low_memory=False)
vis["tp_weeks"] = vis["timepoint"].apply(parse_timepoint_weeks)

# Sentinel for unknown-timepoint patients (T1-T7)
sentinel = -1.0
vis["patient_min_tp"] = vis.groupby("patient_id")["tp_weeks"].transform("min")
vis["patient_max_tp"] = vis.groupby("patient_id")["tp_weeks"].transform("max")
vis.loc[vis["patient_min_tp"].isna(), "patient_min_tp"] = sentinel
vis.loc[vis["patient_max_tp"].isna(), "patient_max_tp"] = sentinel
vis["tp_weeks"] = vis["tp_weeks"].fillna(sentinel)

print(f"  {len(vis):,} observations, {vis['patient_id'].nunique()} patients")

# -- 2. First-tp integration sites -------------------------------------------
first_sites = (
    vis[vis["tp_weeks"] == vis["patient_min_tp"]]
    [["patient_id", "chrom", "site_pos"]]
    .drop_duplicates()
)
print(f"  First-tp confirmed sites: {len(first_sites):,}")

# -- 3. Last-tp label ---------------------------------------------------------
last_agg = (
    vis[vis["tp_weeks"] == vis["patient_max_tp"]]
    .groupby(["patient_id", "chrom", "site_pos"])["clonal_abundance"]
    .max()
    .reset_index()
    .rename(columns={"clonal_abundance": "last_tp_abundance"})
)

# -- 4. Load gene mappings ----------------------------------------------------
print("Loading gene mappings...")
usecols = ["chrom", "site_pos", "nearest_gene_name"] + GENOMIC_COLS
site_gene = (
    pd.read_csv(TRAIN_PATH, low_memory=False, usecols=usecols)
    .drop_duplicates(subset=["chrom", "site_pos"])
)

# -- 5. Map first-tp sites to genes ------------------------------------------
first_mapped = first_sites.merge(
    site_gene[["chrom", "site_pos", "nearest_gene_name"]],
    on=["chrom", "site_pos"], how="inner"
)
patient_gene = first_mapped[["patient_id", "nearest_gene_name"]].drop_duplicates()

gene_chrom = (
    site_gene[["nearest_gene_name", "chrom"]]
    .drop_duplicates(subset="nearest_gene_name", keep="first")
)
patient_gene = patient_gene.merge(gene_chrom, on="nearest_gene_name", how="left")

# -- 6. Compute growth rate per (patient, gene) -------------------------------
print("Computing growth rates...")

# All observations of confirmed-integration sites, mapped to genes
all_obs = vis.merge(first_sites, on=["patient_id", "chrom", "site_pos"], how="inner")
all_obs = all_obs.merge(
    site_gene[["chrom", "site_pos", "nearest_gene_name"]],
    on=["chrom", "site_pos"], how="inner"
)
all_obs["log_abundance"] = np.log1p(all_obs["clonal_abundance"])

def _growth(grp):
    tp = grp["tp_weeks"].values
    ab = grp["log_abundance"].values
    # Only use timepoints with real week values (not sentinel)
    valid = tp > 0
    n_tp = int(valid.sum())
    if n_tp >= 2:
        slope = float(np.polyfit(tp[valid], ab[valid], 1)[0])
    elif n_tp == 1:
        slope = 0.0
    else:
        slope = 0.0
    return pd.Series({"growth_rate": slope, "n_timepoints": n_tp})

print("  Fitting per-(patient, gene) trajectories (may take ~30s)...")
growth = (
    all_obs
    .groupby(["patient_id", "nearest_gene_name"], sort=False)
    .apply(_growth, include_groups=False)
    .reset_index()
)
print(f"  Growth rates computed for {len(growth):,} (patient, gene) pairs")
print(f"  growth_rate range: {growth['growth_rate'].min():.3f} - {growth['growth_rate'].max():.3f}")
print(f"  Pairs with >= 2 timepoints: {(growth['n_timepoints'] >= 2).sum():,} "
      f"({(growth['n_timepoints'] >= 2).mean()*100:.1f}%)")

# -- 7. Build label -----------------------------------------------------------
last_mapped = last_agg.merge(
    site_gene[["chrom", "site_pos", "nearest_gene_name"]],
    on=["chrom", "site_pos"], how="inner"
)
last_gene = (
    last_mapped
    .groupby(["patient_id", "nearest_gene_name"])["last_tp_abundance"]
    .max()
    .reset_index()
)

df = patient_gene.merge(last_gene, on=["patient_id", "nearest_gene_name"], how="left")
df["last_tp_abundance"] = df["last_tp_abundance"].fillna(0)

# v3: remove entries where the clone was not detectable at last timepoint.
# Keeps only (patient, gene) pairs where integration persisted — the model
# then learns what predicts expansion magnitude, not just persistence.
before = len(df)
df = df[df["last_tp_abundance"] > 0].copy()
print(f"  Removed {before - len(df):,} vanished clones ({(before - len(df))/before*100:.1f}%), "
      f"kept {len(df):,} persistent integrations")

# -- 8. Join growth rate + genomic features ----------------------------------
df = df.merge(growth, on=["patient_id", "nearest_gene_name"], how="left")
df["growth_rate"]    = df["growth_rate"].fillna(0.0)
df["n_timepoints"]   = df["n_timepoints"].fillna(1).astype(int)

df["label_raw"] = np.log1p(df["last_tp_abundance"])

# Within-patient percentile rank: removes sequencing depth differences between
# patients. Label = 0.0 (lowest abundance for this patient) to 1.0 (highest).
# The model now learns which genomic features predict being a top-ranked clone
# within a patient, which is purely biological signal.
df["label"] = df.groupby("patient_id")["label_raw"].rank(pct=True)

gene_features = (
    site_gene.drop_duplicates(subset="nearest_gene_name", keep="first")
    [["nearest_gene_name"] + GENOMIC_COLS]
)
df = df.merge(gene_features, on="nearest_gene_name", how="left")
df["patient_id_enc"] = pd.Categorical(df["patient_id"]).codes

# -- Join external IS features -----------------------------------------------
ext_path = os.path.join(ROOT, "data", "external_is_features.csv")
if os.path.exists(ext_path):
    ext = pd.read_csv(ext_path)
    df = df.merge(ext, on="nearest_gene_name", how="left")
    for col in ["ext_n_patients", "ext_n_datasets", "ext_n_unique_sites", "ext_max_abundance"]:
        df[col] = df[col].fillna(0)
    print(f"\nExternal IS features joined: {(df['ext_n_patients'] > 0).sum():,} rows with external data")
else:
    print("\nWarning: data/external_is_features.csv not found — skipping external features")

CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
chrom_map = {c: i for i, c in enumerate(CHROM_ORDER)}
df["chrom_int"] = df["chrom"].map(chrom_map)

# -- 9. Report & save --------------------------------------------------------
print(f"\nLabel range: {df['label'].min():.3f} - {df['label'].max():.3f}")
print(f"Label std:   {df['label'].std():.3f}")
print(f"Label = 0 (clone gone): {(df['label']==0).sum():,} / {len(df):,} "
      f"({(df['label']==0).mean()*100:.1f}%)")

print(f"\nCorrelation of growth_rate with label:")
r = df["growth_rate"].corr(df["label"])
print(f"  r = {r:.3f}")

print(f"\nCorrelation by n_timepoints group:")
for n in sorted(df["n_timepoints"].unique()):
    sub = df[df["n_timepoints"] == n]
    r = sub["growth_rate"].corr(sub["label"])
    print(f"  n_tp={n}: n={len(sub):,}  r(growth_rate, label)={r:.3f}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} rows x {len(df.columns)} cols -> patient_gene_features_v2.csv")
