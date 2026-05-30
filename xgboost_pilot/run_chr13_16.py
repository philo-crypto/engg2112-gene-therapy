#!/usr/bin/env python3
"""
Chr13-16 pilot: LightGBM with temporal features.
Chromosome-stratified split: train=chr13+14, val=chr15, test=chr16.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(PILOT_DIR, "..", "training_data.csv")

FEATURE_COLS = [
    # Genomic position features
    "dist_to_nearest_tss",
    "in_gene_body",
    "dist_to_nearest_cancer_tss",
    "gene_length",
    "in_repeat",
    # CpG features
    "cpg_island_count",
    "cpg_nearest_dist",
    "cpg_mean_obs_exp",
    # ENCODE regulatory elements
    "encode_ccre_count",
    "encode_promoter_count",
    "encode_enhancer_count",
    "encode_ctcf_count",
    # Open Targets cancer annotations
    "ot_cancer_hallmark_count",
    "ot_disease_count",
    "ot_is_cancer_driver",
    # Temporal / clonal dynamics (new)
    "n_samples_detected",
    "growth_rate",
    "abundance_fold_change",
]

TRAIN_CHROMS = ["chr13", "chr14"]
VAL_CHROM   = "chr15"
TEST_CHROM  = "chr16"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("STEP 1: Load chr13-16")

df_all = pd.read_csv(DATA_PATH, low_memory=False)
df = df_all[df_all["chrom"].isin(TRAIN_CHROMS + [VAL_CHROM, TEST_CHROM])].copy()
print(f"Total rows: {len(df):,}  |  Cols: {df.shape[1]}")
print(f"Patients: {sorted(df['patient_id'].unique())}")
print(f"Per-chrom breakdown:")
for chrom, n in df["chrom"].value_counts().sort_index().items():
    print(f"  {chrom}: {n:,} rows")

# ── 2. Inspect ────────────────────────────────────────────────────────────────

print("STEP 2: Inspect")

print("\nLabel distribution:")
print(df["label"].describe().round(4).to_string())
print(f"\nTop 5% high-risk sites: {(df['label'] > df['label'].quantile(0.95)).sum():,} / {len(df):,}")

print("\nTemporal features summary:")
for col in ["n_samples_detected", "growth_rate", "abundance_fold_change"]:
    s = df[col]
    print(f"  {col}: mean={s.mean():.3f}  std={s.std():.3f}  "
          f"min={s.min():.3f}  max={s.max():.3f}  nulls={s.isna().sum()}")

print("\nNull counts per feature:")
null_counts = df[FEATURE_COLS].isnull().sum()
print(null_counts[null_counts > 0].to_string() if null_counts.any() else "  None")

# ── 3. Clean ──────────────────────────────────────────────────────────────────

print("STEP 3: Clean")

before = len(df)

df = df.dropna(subset=["label", "nearest_gene_name"])
df["cpg_mean_obs_exp"]    = df["cpg_mean_obs_exp"].fillna(0.0)
df["growth_rate"]         = df["growth_rate"].fillna(0.0)
df["abundance_fold_change"] = df["abundance_fold_change"].fillna(0.0)

# Clip distances
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].clip(upper=250_000_000)

# Clip growth_rate at 1st/99th percentile — extreme outliers from noisy 2-point fits
lo, hi = df["growth_rate"].quantile(0.01), df["growth_rate"].quantile(0.99)
df["growth_rate"] = df["growth_rate"].clip(lower=lo, upper=hi)

print(f"Rows: {before:,} -> {len(df):,}  (dropped {before - len(df)})")
print(f"growth_rate clipped to [{lo:.1f}, {hi:.1f}]")
print(f"Remaining nulls: {df[FEATURE_COLS].isnull().sum().sum()}")
print(f"  (dist_to_nearest_cancer_tss NaN: {df['dist_to_nearest_cancer_tss'].isna().sum()} — handled by LightGBM)")

# ── 4. Chromosome-stratified split ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Chromosome-stratified split")
print("=" * 60)
print("  Train: chr13 + chr14  (optimise on these)")
print("  Val:   chr15          (early stopping)")
print("  Test:  chr16          (held-out, never seen during training)")

train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[FEATURE_COLS].values, train_df["label"].values
X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df["label"].values
X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["label"].values

print(f"\n  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

for name, Xs, ys in [("train", X_train, y_train),
                     ("val",   X_val,   y_val),
                     ("test",  X_test,  y_test)]:
    out = pd.DataFrame(Xs, columns=FEATURE_COLS)
    out["label"] = ys
    out.to_csv(os.path.join(PILOT_DIR, f"chr13_16_{name}.csv"), index=False)
    print(f"  Saved chr13_16_{name}.csv")

# ── 5. LightGBM ───────────────────────────────────────────────────────────────

print("STEP 5: LightGBM")


dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain)

params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       63,        # max leaf nodes per tree (more = more complex)
    "learning_rate":    0.05,
    "feature_fraction": 0.8,       # fraction of features sampled per tree
    "bagging_fraction": 0.8,       # fraction of rows sampled per tree
    "bagging_freq":     5,         # apply bagging every 5 iterations
    "min_child_samples": 20,       # min rows per leaf (regularisation)
    "lambda_l1":        0.1,       # L1 regularisation
    "lambda_l2":        0.1,       # L2 regularisation
    "verbose":          -1,
    "seed":             42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=50),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=callbacks,
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

print("STEP 6: Evaluation")


for split_name, Xs, ys in [("Train (chr13+14)", X_train, y_train),
                            ("Val   (chr15)",    X_val,   y_val),
                            ("Test  (chr16)",    X_test,  y_test)]:
    pred   = model.predict(Xs)
    r2     = r2_score(ys, pred)
    rmse   = np.sqrt(mean_squared_error(ys, pred))
    mae    = mean_absolute_error(ys, pred)ffy
    thresh = np.percentile(ys, 95)
    mask   = ys >= thresh
    high_r2 = r2_score(ys[mask], pred[mask]) if mask.sum() > 0 else float("nan")
    print(f"\n  {split_name}:")
    print(f"    R2:          {r2:.4f}  (target >= 0.65)")
    print(f"    RMSE:        {rmse:.4f}")
    print(f"    MAE:         {mae:.4f}")
    print(f"    R2 top-5pct: {high_r2:.4f}  (high-risk sensitivity)")

# ── 7. Feature importance
print("STEP 7: Feature Importance (gain)")

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=False)
for feat, score in importance.items():
    marker = " <-- temporal" if feat in ("n_samples_detected", "growth_rate", "abundance_fold_change") else ""
    print(f"  {feat}: {score:.0f}{marker}")

model.save_model(os.path.join(PILOT_DIR, "chr13_16_lgbm.txt"))
print("\nModel saved -> xgboost_pilot/chr13_16_lgbm.txt")
print("\nDone.")


import json
with open(os.path.join(PILOT_DIR, "chr13_16_lgbm.json"), "w") as f:
    json.dump(model.dump_model(), f, indent=2)
print("Model also saved -> xgboost_pilot/chr13_16_lgbm.json")

