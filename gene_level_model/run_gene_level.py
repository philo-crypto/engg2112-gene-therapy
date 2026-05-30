#!/usr/bin/env python3
"""
Gene-level LightGBM model — Option C: insertion site frequency prediction.
One row per gene; label = log1p(n_site_obs) — how many times any patient
inserted near this gene. This is driven by chromatin state (open chromatin,
active promoters) and is directly predictable from ENCODE/CpG features.
Chromosome-stratified split: test=chr1, val=chr2, train=all others.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PILOT_DIR, "gene_features.csv")

FEATURE_COLS = [
    # Truly gene-level features only (same value for every site near the gene)
    "dist_to_nearest_cancer_tss",
    "gene_length",
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
    # GTEx expression features
    "gtex_whole_blood_tpm",
    "gtex_spleen_tpm",
    "gtex_immune_mean_tpm",
]

LABEL_COL = "insertion_freq"  # log1p(n_site_obs) — insertion frequency per gene

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# -- 1. Load ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load gene features")
print("=" * 60)
df = pd.read_csv(DATA_PATH, low_memory=False)
df["insertion_freq"] = np.log1p(df["n_site_obs"])
print(f"Total genes: {len(df):,}  |  Cols: {df.shape[1]}")
print(f"Label (log1p insertion count) range: {df['insertion_freq'].min():.3f} - {df['insertion_freq'].max():.3f}")

counts = df["chrom"].value_counts().reindex(ALL_CHROMS).fillna(0).astype(int)
print(f"\nPer-chrom gene counts:")
for chrom, n in counts.items():
    role = " [TEST]" if chrom == TEST_CHROM else " [VAL]" if chrom == VAL_CHROM else ""
    print(f"  {chrom}: {n:,}{role}")

# -- 2. Clean -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Clean")
print("=" * 60)
before = len(df)
df = df.dropna(subset=["n_site_obs"]).copy()
df["cpg_mean_obs_exp"]           = df["cpg_mean_obs_exp"].fillna(0.0)
df["ot_cancer_hallmark_count"]   = df["ot_cancer_hallmark_count"].fillna(0.0)
df["ot_disease_count"]           = df["ot_disease_count"].fillna(0.0)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].clip(upper=250_000_000)
print(f"Genes: {before:,} -> {len(df):,}  (dropped {before - len(df)})")
print(f"Remaining nulls: {df[FEATURE_COLS].isnull().sum().sum()}")

# -- 3. Split -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Chromosome-stratified split")
print("=" * 60)
avail_feats = [f for f in FEATURE_COLS if f in df.columns]

train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[avail_feats].values, train_df[LABEL_COL].values
X_val,   y_val   = val_df[avail_feats].values,   val_df[LABEL_COL].values
X_test,  y_test  = test_df[avail_feats].values,  test_df[LABEL_COL].values

print(f"Train ({len(TRAIN_CHROMS)} chroms): {len(X_train):,} genes")
print(f"Val   (chr2):            {len(X_val):,} genes")
print(f"Test  (chr1):            {len(X_test):,} genes")
print(f"Features: {len(avail_feats)}")

# -- 4. LightGBM --------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: LightGBM")
print("=" * 60)

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail_feats)
dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=avail_feats, reference=dtrain)

params = {
    "objective":         "regression",
    "metric":            "rmse",
    "max_depth":         6,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 10,
    "reg_lambda":        0.1,
    "seed":              42,
    "verbose":           -1,
}

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]
model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=callbacks,
)

# -- 5. Evaluate --------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

for split_name, X, ys, split_df in [
    ("Train",        X_train, y_train, train_df),
    ("Val   (chr2)", X_val,   y_val,   val_df),
    ("Test  (chr1)", X_test,  y_test,  test_df),
]:
    pred    = model.predict(X)
    r2      = r2_score(ys, pred)
    rmse    = np.sqrt(mean_squared_error(ys, pred))
    mae     = mean_absolute_error(ys, pred)
    thresh  = np.percentile(ys, 95)
    mask    = ys >= thresh
    high_r2 = r2_score(ys[mask], pred[mask]) if mask.sum() > 0 else float("nan")
    print(f"\n  {split_name}:")
    print(f"    R2:          {r2:.4f}  (target >= 0.65)")
    print(f"    RMSE:        {rmse:.4f}")
    print(f"    MAE:         {mae:.4f}")
    print(f"    R2 top-5pct: {high_r2:.4f}  (high-risk gene sensitivity)")
    print(f"    n genes:     {len(ys):,}")

    # Top 10 highest-risk genes on test set
    if "chr1" in split_name:
        top_idx = np.argsort(pred)[-10:][::-1]
        print(f"\n  Top 10 predicted high-risk genes (chr1):")
        for i in top_idx:
            gene = split_df["nearest_gene_name"].values[i]
            print(f"    {gene}: pred={pred[i]:.3f}  actual={ys[i]:.3f}")

# -- 6. Feature importance ----------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (gain)")
print("=" * 60)
scores = dict(zip(avail_feats, model.feature_importance(importance_type="gain")))
for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {score:.0f}")

# -- 7. Save ------------------------------------------------------------------
model.save_model(os.path.join(PILOT_DIR, "gene_level_lgbm.txt"))
with open(os.path.join(PILOT_DIR, "gene_level_lgbm.json"), "w") as f:
    json.dump(model.dump_model(), f, indent=2)
print("\nModel saved -> gene_level_model/gene_level_lgbm.txt")
print("           -> gene_level_model/gene_level_lgbm.json")
print("\nDone.")
