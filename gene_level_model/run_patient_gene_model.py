#!/usr/bin/env python3
"""
Patient-gene level LightGBM model.
Rows = (patient, gene) pairs with confirmed integration at time 1.
Label = log1p(abundance at last timepoint; 0 if clone contracted).
Non-integrated genes excluded entirely.
Chromosome-stratified split: test=chr1, val=chr2, train=all others.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PILOT_DIR, "patient_gene_features.csv")

FEATURE_COLS = [
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
    "patient_id_enc",
]

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# -- 1. Load ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load patient-gene features")
print("=" * 60)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Total (patient, gene) pairs: {len(df):,}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Unique genes:    {df['nearest_gene_name'].nunique():,}")
print(f"Label range:     {df['label'].min():.3f} - {df['label'].max():.3f}")
print(f"Label = 0 (clone gone at last tp): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")

counts = df["chrom"].value_counts().reindex(ALL_CHROMS).fillna(0).astype(int)
print(f"\nPer-chrom pair counts:")
for chrom, n in counts.items():
    if n > 0:
        role = " [TEST]" if chrom == TEST_CHROM else " [VAL]" if chrom == VAL_CHROM else ""
        print(f"  {chrom}: {n:,}{role}")

# -- 2. Clean -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Clean")
print("=" * 60)
before = len(df)
df = df.dropna(subset=["label"]).copy()
df["cpg_mean_obs_exp"]           = df["cpg_mean_obs_exp"].fillna(0.0)
df["ot_cancer_hallmark_count"]   = df["ot_cancer_hallmark_count"].fillna(0.0)
df["ot_disease_count"]           = df["ot_disease_count"].fillna(0.0)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].fillna(
    df["dist_to_nearest_cancer_tss"].median()
).clip(upper=250_000_000)
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
print(f"Pairs: {before:,} -> {len(df):,}  (dropped {before - len(df)})")
print(f"Remaining nulls: {df[FEATURE_COLS].isnull().sum().sum()}")

# -- 3. Split -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Chromosome-stratified split")
print("=" * 60)
avail_feats = [f for f in FEATURE_COLS if f in df.columns]

train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[avail_feats].values, train_df["label"].values
X_val,   y_val   = val_df[avail_feats].values,   val_df["label"].values
X_test,  y_test  = test_df[avail_feats].values,  test_df["label"].values

print(f"Train ({len(TRAIN_CHROMS)} chroms): {len(X_train):,} pairs")
print(f"Val   (chr2):            {len(X_val):,} pairs")
print(f"Test  (chr1):            {len(X_test):,} pairs")
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
    "max_depth":         5,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 5,
    "reg_lambda":        0.1,
    "seed":              42,
    "verbose":           -1,
}

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(25)]
model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
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
    print(f"    R2 top-5pct: {high_r2:.4f}  (high-abundance clone sensitivity)")
    print(f"    n pairs:     {len(ys):,}")

    if "chr1" in split_name and len(ys) > 0:
        top_n = min(10, len(ys))
        top_idx = np.argsort(pred)[-top_n:][::-1]
        print(f"\n  Top {top_n} predicted high-risk (patient, gene) pairs on chr1:")
        for i in top_idx:
            gene = split_df["nearest_gene_name"].values[i]
            pat  = split_df["patient_id"].values[i]
            print(f"    {gene} [{pat}]: pred={pred[i]:.3f}  actual={ys[i]:.3f}")

# -- 6. Feature importance ----------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (gain)")
print("=" * 60)
scores = dict(zip(avail_feats, model.feature_importance(importance_type="gain")))
for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {score:.0f}")

# -- 7. Save ------------------------------------------------------------------
model.save_model(os.path.join(PILOT_DIR, "patient_gene_lgbm.txt"))
with open(os.path.join(PILOT_DIR, "patient_gene_lgbm.json"), "w") as f:
    json.dump(model.dump_model(), f, indent=2)
print("\nModel saved -> gene_level_model/patient_gene_lgbm.txt")
print("\nDone.")
