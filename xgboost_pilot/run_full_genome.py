#!/usr/bin/env python3
"""
Full-genome LightGBM model.
Train on chr3-chr22 + chrX + chrY, validate on chr2, test on chr1.
Label: log1p(max_raw_abundance) as per original proposal.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(PILOT_DIR, "..", "training_data.csv")

FEATURE_COLS = [
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
    "n_samples_detected",
    "growth_rate",
    "abundance_fold_change",
    "n_patients_with_site",
    "patient_id_enc",
]

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# -- 1. Load ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load full genome")
print("=" * 60)
df_all = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Total rows: {len(df_all):,}  |  Cols: {df_all.shape[1]}")

missing_chroms = [c for c in ALL_CHROMS if c not in df_all["chrom"].unique()]
if missing_chroms:
    print(f"WARNING: missing chromosomes: {missing_chroms}")

missing_feats = [f for f in FEATURE_COLS if f not in df_all.columns]
if missing_feats:
    print(f"WARNING: missing feature columns: {missing_feats}")
    print("  Re-run build_training_data.py to regenerate training_data.csv")

print(f"\nPer-chrom row counts:")
counts = df_all["chrom"].value_counts().reindex(ALL_CHROMS).fillna(0).astype(int)
for chrom, n in counts.items():
    role = " [TEST]" if chrom == TEST_CHROM else " [VAL]" if chrom == VAL_CHROM else ""
    print(f"  {chrom}: {n:,}{role}")

# -- 2. Clean -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Clean")
print("=" * 60)
before = len(df_all)
df = df_all.dropna(subset=["label", "nearest_gene_name"]).copy()
df["cpg_mean_obs_exp"]      = df["cpg_mean_obs_exp"].fillna(0.0)
df["growth_rate"]           = df["growth_rate"].fillna(0.0)
df["abundance_fold_change"] = df["abundance_fold_change"].fillna(0.0)
df["n_patients_with_site"]  = df["n_patients_with_site"].fillna(1).astype(int)
df["patient_id_enc"]        = df["patient_id_enc"].fillna(-1).astype(int)
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].clip(upper=250_000_000)
lo, hi = df["growth_rate"].quantile(0.01), df["growth_rate"].quantile(0.99)
df["growth_rate"] = df["growth_rate"].clip(lower=lo, upper=hi)
print(f"Rows: {before:,} -> {len(df):,}  (dropped {before - len(df)})")
print(f"growth_rate clipped to [{lo:.1f}, {hi:.1f}]")
print(f"Remaining nulls in features: {df[FEATURE_COLS].isnull().sum().sum()}")
print(f"Label range: {df['label'].min():.3f} - {df['label'].max():.3f}")

# -- 3. Split -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Chromosome-stratified split")
print("=" * 60)

# Only use features that actually exist in the data
avail_feats = [f for f in FEATURE_COLS if f in df.columns]
if len(avail_feats) < len(FEATURE_COLS):
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    print(f"NOTE: using {len(avail_feats)}/{len(FEATURE_COLS)} features (missing: {missing})")

train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[avail_feats].values, train_df["label"].values
X_val,   y_val   = val_df[avail_feats].values,   val_df["label"].values
X_test,  y_test  = test_df[avail_feats].values,  test_df["label"].values

print(f"Train ({len(TRAIN_CHROMS)} chroms): {len(X_train):,} rows")
print(f"Val   (chr2):            {len(X_val):,} rows")
print(f"Test  (chr1):            {len(X_test):,} rows")
print(f"Features: {len(avail_feats)}")

# -- 4. LightGBM --------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: LightGBM")
print("=" * 60)

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail_feats)
dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=avail_feats, reference=dtrain)

params = {
    "objective":        "regression",
    "metric":           "rmse",
    "max_depth":        6,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_lambda":       0.1,
    "seed":             42,
    "verbose":          -1,
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

for split_name, X, ys in [("Train",        X_train, y_train),
                           ("Val   (chr2)", X_val,   y_val),
                           ("Test  (chr1)", X_test,  y_test)]:
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
    print(f"    R2 top-5pct: {high_r2:.4f}  (high-risk sensitivity)")

# -- 6. Feature importance ----------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (gain)")
print("=" * 60)
temporal_feats = {"n_samples_detected", "growth_rate", "abundance_fold_change",
                  "n_patients_with_site", "patient_id_enc"}
scores = dict(zip(avail_feats, model.feature_importance(importance_type="gain")))
for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
    marker = " <-- temporal/patient" if feat in temporal_feats else ""
    print(f"  {feat}: {score:.0f}{marker}")

# -- 7. Save ------------------------------------------------------------------
model.save_model(os.path.join(PILOT_DIR, "full_genome_lgbm.txt"))
with open(os.path.join(PILOT_DIR, "full_genome_lgbm.json"), "w") as f:
    json.dump(model.dump_model(), f, indent=2)
print("\nModel saved -> xgboost_pilot/full_genome_lgbm.txt")
print("           -> xgboost_pilot/full_genome_lgbm.json")
print("\nDone.")
