#!/usr/bin/env python3
"""
Chr13 pilot: inspect → clean → train/val/test split → XGBoost regression.
Target: label = log1p(max raw read count per patient-site), range ~[1.4, 12]
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

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
]

# ── 1. Load chr13 ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load chr13")
print("=" * 60)
df_all = pd.read_csv(DATA_PATH, low_memory=False)
df = df_all[df_all["chrom"] == "chr13"].copy()
print(f"Rows: {len(df):,}  |  Cols: {df.shape[1]}")
print(f"Patients: {sorted(df['patient_id'].unique())}")

# ── 2. Inspect ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Inspect")
print("=" * 60)
print("\nLabel distribution:")
print(df["label"].describe().round(6).to_string())
high_risk = (df["label"] > df["label"].quantile(0.95)).sum()
print(f"\nTop 5% high-risk sites: {high_risk:,} / {len(df):,}")

print("\nNull counts per feature:")
null_counts = df[FEATURE_COLS].isnull().sum()
print(null_counts[null_counts > 0].to_string() if null_counts.any() else "  None")

print("\nFeature ranges (sample):")
print(df[FEATURE_COLS].describe().loc[["mean","std","min","max"]].round(3).to_string())

# ── 3. Clean ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Clean")
print("=" * 60)
before = len(df)

# Drop rows with null label (shouldn't exist but guard)
df = df.dropna(subset=["label"])

# nearest_gene_name null (171 rows) means no gene on chr13 nearby — drop
df = df.dropna(subset=["nearest_gene_name"])

# cpg_mean_obs_exp null = no CpG overlap → fill with 0 (no CpG signal)
df["cpg_mean_obs_exp"] = df["cpg_mean_obs_exp"].fillna(0.0)

# dist_to_nearest_cancer_tss null = no cancer driver on chromosome
# XGBoost handles NaN natively — leave as-is

# Clip extreme dist values (sanity guard: > 250 Mb would be off-chromosome)
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].clip(upper=250_000_000)

print(f"Rows before: {before:,}  ->  after: {len(df):,}  (dropped {before - len(df)})")
print(f"Remaining nulls in features: {df[FEATURE_COLS].isnull().sum().sum()}")
print(f"  (dist_to_nearest_cancer_tss NaN: {df['dist_to_nearest_cancer_tss'].isna().sum()} — handled by XGBoost)")

# ── 4. Split ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Train / Val / Test split  (70 / 15 / 15)")
print("=" * 60)

X = df[FEATURE_COLS].values
y = df["label"].values

# 70 train, 30 temp → split temp into 50/50 val/test = 15/15 overall
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# Save splits
for name, Xs, ys in [("train", X_train, y_train),
                      ("val",   X_val,   y_val),
                      ("test",  X_test,  y_test)]:
    split_df = pd.DataFrame(Xs, columns=FEATURE_COLS)
    split_df["label"] = ys
    path = os.path.join(PILOT_DIR, f"chr13_{name}.csv")
    split_df.to_csv(path, index=False)
    print(f"  Saved {path.split(os.sep)[-1]}")

# ── 5. XGBoost ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: XGBoost")
print("=" * 60)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_COLS)

params = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "seed":             42,
    "verbosity":        0,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=30,
    verbose_eval=50,
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Evaluation")
print("=" * 60)

for split_name, dm, y_true in [("Train", dtrain, y_train),
                                ("Val",   dval,   y_val),
                                ("Test",  dtest,  y_test)]:
    y_pred = model.predict(dm)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Sensitivity on top 5% (high-risk sites)
    thresh = np.percentile(y_true, 95)
    high_mask = y_true >= thresh
    if high_mask.sum() > 0:
        high_r2 = r2_score(y_true[high_mask], y_pred[high_mask])
    else:
        high_r2 = float("nan")
    print(f"\n  {split_name}:")
    print(f"    R2:            {r2:.4f}  (target >= 0.65)")
    print(f"    RMSE:          {rmse:.6f}")
    print(f"    MAE:           {mae:.6f}")
    print(f"    R2 top-5pct:   {high_r2:.4f}  (high-risk sensitivity)")

# ── 7. Feature importance ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature Importance (gain)")
print("=" * 60)
scores = model.get_score(importance_type="gain")
if scores:
    for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {score:.2f}")
else:
    print("  (no splits recorded — model may not have learned)")

model.save_model(os.path.join(PILOT_DIR, "chr13_xgb.json"))
print(f"\nModel saved -> xgboost_pilot/chr13_xgb.json")
print("\nDone.")
