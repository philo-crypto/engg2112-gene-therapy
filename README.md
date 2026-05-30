# Predicting Insertional Mutagenesis Risk in Gene Therapy

**ENGG2112 — Group 3**

A machine learning pipeline that predicts which genomic integration sites in lentiviral gene therapy are most likely to drive clonal expansion, using genomic and epigenomic features. The goal is to support prospective risk stratification, shifting clinical monitoring from retrospective detection to forward-looking risk prioritisation.

---

## Project Overview

Gene therapy can correct genetic disorders by delivering functional DNA into a patient's cells using a viral vector, but it carries a risk of insertional mutagenesis. When a vector integrates near a proto-oncogene, it can activate cancer-driving genes and cause uncontrolled clonal expansion, as seen in the 2002–2003 SCID-X1 trials where five of twenty patients developed leukaemia.

This project trains a regression model to predict the **within-patient percentile rank of clonal abundance** at the last measured timepoint for each patient-gene pair. The model ranks integration sites by risk, helping clinicians prioritise which sites to monitor most closely.

**Best result:** the stacking ensemble achieved a top-20% lift of **1.95 times** over random selection on a held-out test chromosome.

---

## Dataset

- **Primary data:** Yan et al. (2023) SCID-X1 lentiviral integration site data — 76,402 patient-gene pairs across 17 patients, measured by LAM-PCR sequencing.
- **External integration data:** Calabria et al. (2024) and the Bushman laboratory (2020), pooled across 59 external patients producing 11.8 million integration observations across 14,994 annotated genes.
- **Annotations:** Ensembl, UCSC Genome Browser, ENCODE, Open Targets, and UW Repli-seq.

---

## Features (20 total)

Features are grouped into four categories:

| Category | Examples |
|---|---|
| External integration frequency | `ext_log_unique_sites`, `ext_log_n_patients`, `ext_n_datasets`, `ext_max_abundance` |
| Genomic structure | `dist_to_nearest_tss`, `dist_to_nearest_cancer_tss`, `gene_length`, `in_repeat`, `replication_timing` |
| Epigenomic regulatory | `tcell_h3k27ac_count`, `encode_ccre_count`, `encode_promoter_count`, `encode_enhancer_count`, `encode_ctcf_count`, `cpg_island_count`, `cpg_nearest_dist`, `cpg_mean_obs_exp` |
| Cancer biology | `ot_is_cancer_driver`, `ot_cancer_hallmark_count`, `ot_disease_count` |

The two strongest predictive features were the external integration frequency variables, followed by replication timing as the strongest purely genomic feature.

---

## Models

Five model architectures were evaluated under the same chromosome-stratified split:

| Model | Spearman | Top-20% Lift | R² |
|---|---|---|---|
| **Stacking Ensemble** (best) | 0.381 | **1.95x** | 0.148 |
| LightGBM Regression (baseline) | 0.382 | 1.93x | 0.150 |
| XGBoost Pairwise Ranking | 0.370 | 1.94x | n/a |
| LightGBM LambdaRank | 0.364 | 1.90x | n/a |
| MLP Neural Network | 0.361 | 1.87x | 0.134 |

The stacking ensemble combines LightGBM, XGBoost, and Random Forest via a Ridge meta-learner (weights: LightGBM 0.43, Random Forest 0.34, XGBoost 0.23).

**Validation:** chromosome-stratified split — train on chromosomes 3–22, X, Y; validate on chromosome 2; test on chromosome 1. This tests genuine generalisation to unseen genomic regions.

---

## Repository Structure

```
.
├── README.md                      This file
├── requirements.txt               Python dependencies
├── PIPELINE.md                    Step-by-step pipeline guide
├── MODEL_COMPARISON.md            Detailed model comparison
├── MODEL_REPORT.md                Final model performance report
│
├── download_study_data.py         Downloads Yan et al. 2023 data
├── download_external_is_data.py   Downloads external integration data
├── gene_feature_pipeline.py       Fetches gene features per chromosome
├── run_all_chroms.py              Batch runs the feature pipeline
├── enrich_ot_missing.py           Adds Open Targets annotations
├── enrich_replication_timing.py   Adds replication timing feature
├── enrich_tcell_chromatin.py      Adds T-cell H3K27ac feature
├── build_training_data.py         Joins data and computes labels
│
├── chr1_features.csv ... chrY_features.csv   Per-chromosome feature tables
│
├── xgboost_pilot/                 XGBoost baseline and early LightGBM models
├── gene_level_model/              Gene-level modelling experiments
├── patient_gene_v2/               Patient-gene modelling (earlier version)
└── patient_gene_v3/               Final models: stacking ensemble, LambdaRank,
                                    pairwise ranking, MLP
```
---

## Authors
Group 3 — Tadhg Stuckey, Mahmoud Alhyari, Philopatir Khalil, Xiaowei Sun
