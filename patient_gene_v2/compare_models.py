#!/usr/bin/env python3
"""
Compare LightGBM, XGBoost, Random Forest, and Ridge on patient_gene_features_v2.csv.
Same chromosome-stratified split as run_model.py.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

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
]

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# -- Load & clean -------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["label"]).copy()
df["cpg_mean_obs_exp"]           = df["cpg_mean_obs_exp"].fillna(0.0)
df["ot_cancer_hallmark_count"]   = df["ot_cancer_hallmark_count"].fillna(0.0)
df["ot_disease_count"]           = df["ot_disease_count"].fillna(0.0)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].fillna(
    df["dist_to_nearest_cancer_tss"].median()).clip(upper=250_000_000)
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
lo, hi = df["growth_rate"].quantile(0.01), df["growth_rate"].quantile(0.99)
df["growth_rate"] = df["growth_rate"].clip(lo, hi)

# -- Split --------------------------------------------------------------------
avail = [f for f in FEATURE_COLS if f in df.columns]
train_df = df[df["chrom"].isin(TRAIN_CHROMS)]
val_df   = df[df["chrom"] == VAL_CHROM]
test_df  = df[df["chrom"] == TEST_CHROM]

X_train, y_train = train_df[avail].values, train_df["label"].values
X_val,   y_val   = val_df[avail].values,   val_df["label"].values
X_test,  y_test  = test_df[avail].values,  test_df["label"].values

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}  Features: {len(avail)}")

def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    k = int(len(y_true) * 0.20)
    overlap = len(set(np.argsort(y_pred)[-k:]) & set(np.argsort(y_true)[-k:]))
    lift = (overlap / k) / 0.20
    return {"model": name, "R2": r2, "RMSE": rmse, "Spearman": rho, "Top20_lift": lift}

results = []

# -- 1. LightGBM --------------------------------------------------------------
print("\n[1/4] LightGBM...")
dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail)
dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=avail, reference=dtrain)
params = {
    "objective": "regression", "metric": "rmse",
    "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_samples": 5, "reg_lambda": 0.1,
    "seed": 42, "verbose": -1,
}
lgb_model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
results.append(evaluate("LightGBM", y_test, lgb_model.predict(X_test)))

# -- 2. XGBoost ---------------------------------------------------------------
print("[2/4] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=0.1, seed=42, verbosity=0,
    early_stopping_rounds=50, eval_metric="rmse",
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
results.append(evaluate("XGBoost", y_test, xgb_model.predict(X_test)))

# -- 3. Random Forest ---------------------------------------------------------
print("[3/4] Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=10,
    max_features=0.5, n_jobs=-1, random_state=42,
)
rf_model.fit(X_train, y_train)
results.append(evaluate("RandomForest", y_test, rf_model.predict(X_test)))

# -- 4. Ridge -----------------------------------------------------------------
print("[4/4] Ridge Regression...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
results.append(evaluate("Ridge", y_test, ridge.predict(X_test_s)))

# -- Summary ------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"{'Model':<15} {'R2':>8} {'RMSE':>8} {'Spearman':>10} {'Top-20% lift':>13}")
print("-" * 65)
for r in sorted(results, key=lambda x: -x["Spearman"]):
    print(f"{r['model']:<15} {r['R2']:>8.4f} {r['RMSE']:>8.4f} {r['Spearman']:>10.4f} {r['Top20_lift']:>12.2f}x")
print("=" * 65)
