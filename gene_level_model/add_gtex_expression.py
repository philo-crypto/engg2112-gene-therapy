#!/usr/bin/env python3
"""
Download GTEx v8 gene median TPM and add expression features to gene_features.csv.

Adds three features:
  gtex_whole_blood_tpm  — median TPM in Whole Blood (most relevant to gene therapy)
  gtex_thymus_tpm       — median TPM in Thymus (T-cell development, SCID-X1 relevant)
  gtex_immune_mean_tpm  — mean TPM across Whole Blood, Spleen, Thymus, Lymphocytes

All values are log1p-transformed before saving.
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd

PILOT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(PILOT_DIR, "gtex_median_tpm.gct.gz")
GENE_CSV   = os.path.join(PILOT_DIR, "gene_features.csv")
OUT_CSV    = os.path.join(PILOT_DIR, "gene_features.csv")  # overwrite in place

GTEX_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)

IMMUNE_TISSUES = ["Whole Blood", "Spleen", "Cells - EBV-transformed lymphocytes"]

# -- 1. Download --------------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"GTEx file already cached: {CACHE_PATH}")
else:
    print(f"Downloading GTEx median TPM (~7 MB)...")
    urllib.request.urlretrieve(GTEX_URL, CACHE_PATH)
    print(f"  Saved -> {CACHE_PATH}")

# -- 2. Parse GCT -------------------------------------------------------------
print("Parsing GCT file...")
with gzip.open(CACHE_PATH, "rt") as f:
    f.readline()           # #1.2
    f.readline()           # dimensions
    header = f.readline().strip().split("\t")
    rows = []
    for line in f:
        rows.append(line.strip().split("\t"))

gtex = pd.DataFrame(rows, columns=header)
gtex = gtex.rename(columns={"Name": "ensembl_id", "Description": "gene_name"})
numeric_cols = [c for c in gtex.columns if c not in ("ensembl_id", "gene_name")]
gtex[numeric_cols] = gtex[numeric_cols].astype(float)

print(f"  GTEx: {len(gtex):,} genes x {len(numeric_cols)} tissues")
print(f"  Available immune tissues: {[t for t in IMMUNE_TISSUES if t in gtex.columns]}")

# -- 3. Extract features ------------------------------------------------------
def safe_col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

avail_immune = [t for t in IMMUNE_TISSUES if t in gtex.columns]
print(f"  Using immune tissues: {avail_immune}")

gtex["gtex_whole_blood_tpm"]  = np.log1p(safe_col(gtex, "Whole Blood"))
gtex["gtex_spleen_tpm"]       = np.log1p(safe_col(gtex, "Spleen"))
gtex["gtex_immune_mean_tpm"]  = np.log1p(gtex[avail_immune].mean(axis=1))

expr = gtex[["gene_name", "gtex_whole_blood_tpm",
             "gtex_spleen_tpm", "gtex_immune_mean_tpm"]].copy()
# One row per gene name (GTEx has some duplicates for PAR genes on chrX/Y)
expr = expr.drop_duplicates(subset="gene_name", keep="first")

print(f"  Whole Blood TPM range (log1p): "
      f"{gtex['gtex_whole_blood_tpm'].min():.2f} - {gtex['gtex_whole_blood_tpm'].max():.2f}")
print(f"  Spleen TPM range (log1p):      "
      f"{gtex['gtex_spleen_tpm'].min():.2f} - {gtex['gtex_spleen_tpm'].max():.2f}")

# -- 4. Join to gene_features.csv ---------------------------------------------
print("\nJoining to gene_features.csv...")
genes = pd.read_csv(GENE_CSV)
before = len(genes)

# Drop old GTEx columns if re-running
for col in ["gtex_whole_blood_tpm", "gtex_spleen_tpm", "gtex_immune_mean_tpm"]:
    if col in genes.columns:
        genes.drop(columns=[col], inplace=True)

genes = genes.merge(expr, left_on="nearest_gene_name", right_on="gene_name", how="left")
genes.drop(columns=["gene_name"], inplace=True, errors="ignore")

matched = genes["gtex_whole_blood_tpm"].notna().sum()
print(f"  {matched:,} / {before:,} genes matched to GTEx ({matched/before*100:.1f}%)")
print(f"  Whole Blood coverage check (unmatched genes): "
      f"{genes['gtex_whole_blood_tpm'].isna().sum():,} NaNs -> will be filled with 0")

genes["gtex_whole_blood_tpm"]  = genes["gtex_whole_blood_tpm"].fillna(0.0)
genes["gtex_thymus_tpm"]       = genes["gtex_thymus_tpm"].fillna(0.0)
genes["gtex_immune_mean_tpm"]  = genes["gtex_immune_mean_tpm"].fillna(0.0)

genes.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(genes):,} rows x {len(genes.columns)} cols -> gene_features.csv")
print(f"New columns: gtex_whole_blood_tpm, gtex_thymus_tpm, gtex_immune_mean_tpm")

# -- 5. Quick correlation check -----------------------------------------------
print("\n=== Correlation of new features with insertion frequency ===")
genes["insertion_freq"] = np.log1p(genes["n_site_obs"])
for col in ["gtex_whole_blood_tpm", "gtex_spleen_tpm", "gtex_immune_mean_tpm"]:
    r = genes[col].corr(genes["insertion_freq"])
    print(f"  {col}: r={r:.3f}")
