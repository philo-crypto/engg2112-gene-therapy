#!/usr/bin/env python3
"""
Patient-gene model v2 — genomic features + growth_rate.
Chromosome-stratified split: test=chr1, val=chr2, train=all others.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, recall_score, precision_score, f1_score)

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PILOT_DIR, "patient_gene_features_v2.csv")

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
    "ext_log_unique_sites",
    "ext_log_n_patients",
    "ext_n_datasets",
    "ext_max_abundance",
    "tcell_h3k27ac_count",
    "replication_timing",
]

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# -- 1. Load ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load")
print("=" * 60)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"(patient, gene) pairs: {len(df):,}  |  patients: {df['patient_id'].nunique()}")
print(f"Label range: {df['label'].min():.3f} - {df['label'].max():.3f}  "
      f"std: {df['label'].std():.3f}")
print(f"growth_rate range: {df['growth_rate'].min():.4f} - {df['growth_rate'].max():.4f}")

# -- 2. Clean -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Clean")
print("=" * 60)
df = df.dropna(subset=["label"]).copy()
df["cpg_mean_obs_exp"]           = df["cpg_mean_obs_exp"].fillna(0.0)
df["ot_cancer_hallmark_count"]   = df["ot_cancer_hallmark_count"].fillna(0.0)
df["ot_disease_count"]           = df["ot_disease_count"].fillna(0.0)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].fillna(
    df["dist_to_nearest_cancer_tss"].median()).clip(upper=250_000_000)
df["tcell_h3k27ac_count"] = df["tcell_h3k27ac_count"].fillna(0.0) if "tcell_h3k27ac_count" in df.columns else 0.0
df["replication_timing"]  = df["replication_timing"].fillna(df["replication_timing"].median()) if "replication_timing" in df.columns else 0.0
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)

# Clip growth_rate outliers
lo, hi = df["growth_rate"].quantile(0.01), df["growth_rate"].quantile(0.99)
df["growth_rate"] = df["growth_rate"].clip(lo, hi)
print(f"growth_rate clipped to [{lo:.4f}, {hi:.4f}]")
print(f"Remaining nulls: {df[FEATURE_COLS].isnull().sum().sum()}")

# -- 3. Split -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Split")
print("=" * 60)
avail = [f for f in FEATURE_COLS if f in df.columns]
train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[avail].values, train_df["label"].values
X_val,   y_val   = val_df[avail].values,   val_df["label"].values
X_test,  y_test  = test_df[avail].values,  test_df["label"].values

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}  Features: {len(avail)}")

# -- 4. LightGBM + hyperparameter tuning -------------------------------------
print("\n" + "=" * 60)
print("STEP 4: LightGBM (hyperparameter search)")
print("=" * 60)
dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail)
dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=avail, reference=dtrain)

param_grid = [
    {"max_depth": d, "num_leaves": nl, "min_child_samples": mc, "reg_lambda": rl}
    for d  in (5, 7)
    for nl in (31, 63, 127)
    for mc in (5, 20)
    for rl in (0.1, 1.0)
]

base_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "seed": 42, "verbose": -1,
}

best_val_rmse, best_params, best_rounds = float("inf"), None, 0
print(f"  Searching {len(param_grid)} configs...")
for p in param_grid:
    params = {**base_params, **p}
    cb = lgb.train(params, dtrain, num_boost_round=500,
                   valid_sets=[dval], valid_names=["val"],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    score = cb.best_score["val"]["rmse"]
    if score < best_val_rmse:
        best_val_rmse = score
        best_params   = params
        best_rounds   = cb.best_iteration

print(f"  Best val RMSE: {best_val_rmse:.6f}  params: {best_params}")
print(f"  Retraining with best params ({best_rounds} rounds)...")
model = lgb.train(best_params, dtrain, num_boost_round=best_rounds,
                  valid_sets=[dtrain, dval], valid_names=["train", "val"],
                  callbacks=[lgb.log_evaluation(0)])

# -- 5. Evaluate --------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

for split_name, X, ys in [("Train",        X_train, y_train),
                           ("Val   (chr2)", X_val,   y_val),
                           ("Test  (chr1)", X_test,  y_test)]:
    pred = model.predict(X)
    r2   = r2_score(ys, pred)
    rmse = np.sqrt(mean_squared_error(ys, pred))
    mae  = mean_absolute_error(ys, pred)
    rho, _ = spearmanr(ys, pred)

    # Binary at median threshold
    thresh  = np.percentile(ys, 50)
    yb_true = (ys >= thresh).astype(int)
    yb_pred = (pred >= thresh).astype(int)
    acc  = accuracy_score(yb_true, yb_pred)
    rec  = recall_score(yb_true, yb_pred, zero_division=0)
    prec = precision_score(yb_true, yb_pred, zero_division=0)
    f1   = f1_score(yb_true, yb_pred, zero_division=0)

    # Rank lift at top 20%
    k = int(len(ys) * 0.20)
    overlap = len(set(np.argsort(pred)[-k:]) & set(np.argsort(ys)[-k:]))
    lift = (overlap / k) / 0.20

    print(f"\n  {split_name}:")
    print(f"    R2:            {r2:.4f}")
    print(f"    RMSE:          {rmse:.4f}   MAE: {mae:.4f}")
    print(f"    Spearman rho:  {rho:.4f}")
    print(f"    Accuracy:      {acc:.3f}  (median split)")
    print(f"    Recall:        {rec:.3f}  Precision: {prec:.3f}  F1: {f1:.3f}")
    print(f"    Top-20% lift:  {lift:.2f}x  (random = 1.0x)")

# -- 6. Feature importance ----------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (gain)")
print("=" * 60)
scores = dict(zip(avail, model.feature_importance(importance_type="gain")))
for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {score:.0f}")

# -- 7. Save ------------------------------------------------------------------
model.save_model(os.path.join(PILOT_DIR, "patient_gene_v2_lgbm.txt"))
with open(os.path.join(PILOT_DIR, "patient_gene_v2_lgbm.json"), "w") as f:
    json.dump(model.dump_model(), f, indent=2)
print("\nModel saved -> patient_gene_v2/patient_gene_v2_lgbm.txt")
print("Done.")
