#!/usr/bin/env python3
"""
Four alternative models vs the LightGBM regression baseline.

1. LightGBM LambdaRank  — ranking objective, within-patient groups (NDCG)
2. XGBoost Pairwise     — pairwise ranking within patient groups
3. MLP Neural Network   — sklearn MLPRegressor, non-linear feature interactions
4. Stacking Ensemble    — LightGBM + XGBoost + RF predictions → Ridge meta-learner

Same chromosome-stratified split as run_model.py: test=chr1, val=chr2, train=rest.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PILOT_DIR, "patient_gene_features_v2.csv")

FEATURE_COLS = [
    "dist_to_nearest_tss", "dist_to_nearest_cancer_tss", "gene_length",
    "in_repeat", "cpg_island_count", "cpg_nearest_dist", "cpg_mean_obs_exp",
    "encode_ccre_count", "encode_promoter_count", "encode_enhancer_count",
    "encode_ctcf_count", "ot_cancer_hallmark_count", "ot_disease_count",
    "ot_is_cancer_driver", "ext_log_unique_sites", "ext_log_n_patients",
    "ext_n_datasets", "ext_max_abundance", "tcell_h3k27ac_count",
    "replication_timing",
]

ALL_CHROMS   = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TEST_CHROM   = "chr1"
VAL_CHROM    = "chr2"
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in (TEST_CHROM, VAL_CHROM)]

# ── Load & clean ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["label"]).copy()
df["cpg_mean_obs_exp"]           = df["cpg_mean_obs_exp"].fillna(0.0)
df["ot_cancer_hallmark_count"]   = df["ot_cancer_hallmark_count"].fillna(0.0)
df["ot_disease_count"]           = df["ot_disease_count"].fillna(0.0)
df["dist_to_nearest_cancer_tss"] = df["dist_to_nearest_cancer_tss"].fillna(
    df["dist_to_nearest_cancer_tss"].median()).clip(upper=250_000_000)
df["dist_to_nearest_tss"]        = df["dist_to_nearest_tss"].clip(upper=250_000_000)
df["tcell_h3k27ac_count"]        = df["tcell_h3k27ac_count"].fillna(0.0)
df["replication_timing"]         = df["replication_timing"].fillna(df["replication_timing"].median())
lo, hi = df["growth_rate"].quantile(0.01), df["growth_rate"].quantile(0.99)
df["growth_rate"] = df["growth_rate"].clip(lo, hi)

avail = [f for f in FEATURE_COLS if f in df.columns]

# Sort within each split by patient_id for ranking models
train_df = df[df["chrom"].isin(TRAIN_CHROMS)].sort_values("patient_id").reset_index(drop=True)
val_df   = df[df["chrom"] == VAL_CHROM].sort_values("patient_id").reset_index(drop=True)
test_df  = df[df["chrom"] == TEST_CHROM].sort_values("patient_id").reset_index(drop=True)

X_train, y_train = train_df[avail].values, train_df["label"].values
X_val,   y_val   = val_df[avail].values,   val_df["label"].values
X_test,  y_test  = test_df[avail].values,  test_df["label"].values

# Integer relevance labels (0-4) for ranking objectives
y_train_int = np.clip((y_train * 4).round().astype(int), 0, 4)
y_val_int   = np.clip((y_val   * 4).round().astype(int), 0, 4)

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}  Features: {len(avail)}")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    k = int(len(y_true) * 0.20)
    overlap = len(set(np.argsort(y_pred)[-k:]) & set(np.argsort(y_true)[-k:]))
    lift = (overlap / k) / 0.20
    return {"Model": name, "R2": r2, "RMSE": rmse, "Spearman": rho, "Top20_lift": lift}


results = []

# ── BASELINE: LightGBM Regression (for comparison) ───────────────────────────
print("\n[0/4] LightGBM Regression (baseline)...")
dtrain_r = lgb.Dataset(X_train, label=y_train, feature_name=avail)
dval_r   = lgb.Dataset(X_val,   label=y_val,   feature_name=avail, reference=dtrain_r)
lgb_reg  = lgb.train(
    {"objective": "regression", "metric": "rmse", "max_depth": 7, "learning_rate": 0.05,
     "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
     "min_child_samples": 5, "reg_lambda": 0.1, "seed": 42, "verbose": -1},
    dtrain_r, num_boost_round=500,
    valid_sets=[dval_r], valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
results.append(evaluate("LightGBM Regression", y_test, lgb_reg.predict(X_test)))


# ── MODEL 1: LightGBM LambdaRank ─────────────────────────────────────────────
print("[1/4] LightGBM LambdaRank...")
train_groups = train_df.groupby("patient_id", sort=False).size().values
val_groups   = val_df.groupby("patient_id",   sort=False).size().values
test_groups  = test_df.groupby("patient_id",  sort=False).size().values

dtrain_lr = lgb.Dataset(X_train, label=y_train_int, group=train_groups, feature_name=avail)
dval_lr   = lgb.Dataset(X_val,   label=y_val_int,   group=val_groups,   feature_name=avail,
                         reference=dtrain_lr)

lgb_rank = lgb.train(
    {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20],
     "max_depth": 7, "learning_rate": 0.05, "num_leaves": 31,
     "subsample": 0.8, "colsample_bytree": 0.8,
     "min_child_samples": 5, "reg_lambda": 0.1, "seed": 42, "verbose": -1},
    dtrain_lr, num_boost_round=500,
    valid_sets=[dval_lr], valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
results.append(evaluate("LightGBM LambdaRank", y_test, lgb_rank.predict(X_test)))


# ── MODEL 2: XGBoost Pairwise Ranking ────────────────────────────────────────
print("[2/4] XGBoost Pairwise Ranking...")
train_qid = pd.Categorical(train_df["patient_id"]).codes
val_qid   = pd.Categorical(val_df["patient_id"]).codes
test_qid  = pd.Categorical(test_df["patient_id"]).codes

xgb_rank = xgb.XGBRanker(
    objective="rank:pairwise", eval_metric="ndcg",
    max_depth=7, learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=0.1,
    early_stopping_rounds=50, seed=42, verbosity=0,
)
xgb_rank.fit(
    X_train, y_train_int, qid=train_qid,
    eval_set=[(X_val, y_val_int)], eval_qid=[val_qid],
    verbose=False,
)
results.append(evaluate("XGBoost Pairwise", y_test, xgb_rank.predict(X_test)))


# ── MODEL 3: MLP Neural Network ──────────────────────────────────────────────
print("[3/4] MLP Neural Network...")
scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_val   = scaler.transform(X_val)
Xs_test  = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)
mlp.fit(Xs_train, y_train)
results.append(evaluate("MLP Neural Network", y_test, mlp.predict(Xs_test)))


# ── MODEL 4: Stacking Ensemble ───────────────────────────────────────────────
print("[4/4] Stacking Ensemble (LightGBM + XGBoost + RF -> Ridge)...")

# Train base learners and collect out-of-fold predictions on val set
lgb_val_pred  = lgb_reg.predict(X_val)
xgb_base = xgb.XGBRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=0.1,
    early_stopping_rounds=50, seed=42, verbosity=0,
)
xgb_base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_pred  = xgb_base.predict(X_val)

rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                            min_samples_leaf=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_val_pred = rf.predict(X_val)

# Stack val predictions → train meta-learner
meta_train = np.column_stack([lgb_val_pred, xgb_val_pred, rf_val_pred])
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y_val)

# Stack test predictions
lgb_test_pred = lgb_reg.predict(X_test)
xgb_test_pred = xgb_base.predict(X_test)
rf_test_pred  = rf.predict(X_test)
meta_test = np.column_stack([lgb_test_pred, xgb_test_pred, rf_test_pred])
stack_pred = meta_model.predict(meta_test)
results.append(evaluate("Stacking Ensemble", y_test, stack_pred))
print(f"  Meta-learner weights: LGB={meta_model.coef_[0]:.3f} "
      f"XGB={meta_model.coef_[1]:.3f} RF={meta_model.coef_[2]:.3f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Model':<28} {'R2':>7} {'RMSE':>7} {'Spearman':>10} {'Top-20% lift':>13}")
print("-" * 70)
for r in sorted(results, key=lambda x: -x["Spearman"]):
    marker = " << best" if r["Spearman"] == max(x["Spearman"] for x in results) else ""
    print(f"{r['Model']:<28} {r['R2']:>7.4f} {r['RMSE']:>7.4f} "
          f"{r['Spearman']:>10.4f} {r['Top20_lift']:>12.2f}x{marker}")
print("=" * 70)
