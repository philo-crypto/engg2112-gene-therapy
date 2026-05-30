# Pipeline Guide — Insertional Mutagenesis Risk Model

Predicts clonal expansion risk at gene therapy integration sites using genomic features + clonal dynamics from Yan et al. (2023) SCID-X1 data.

---

## Prerequisites

Python 3.10+. Install dependencies:

```bash
pip install pandas numpy requests xgboost lightgbm scikit-learn
```

---

## File Overview

| File | What it does |
|---|---|
| `download_study_data.py` | Downloads Yan et al. 2023 VIS data from Zenodo → `data/vis_raw.csv` |
| `gene_feature_pipeline.py` | Fetches gene features for one chromosome from Ensembl / UCSC / ENCODE / Open Targets |
| `run_all_chroms.py` | Runs `gene_feature_pipeline.py` for all chromosomes in batch |
| `build_training_data.py` | Joins VIS data to gene features, computes labels and temporal features → `training_data.csv` |
| `xgboost_pilot/run_pilot.py` | XGBoost baseline on chr13 only |
| `xgboost_pilot/run_chr13_16.py` | LightGBM model — train on chr13+14, validate on chr15, test on chr16 |

---

## Step-by-step

### 1. Download the study data

```bash
python download_study_data.py
```

Downloads the Yan et al. integration site data from Zenodo and saves it to `data/vis_raw.csv`.
Safe to re-run — skips download if the file already exists.

---

### 2. Check which chromosome feature files exist

Feature files (`chr*_features.csv`) for chr1–9 and chr13–22, X, Y are already in the repo.

**Missing: chr10, chr11, chr12.** To generate them:

```bash
python gene_feature_pipeline.py --chrom chr10 --output chr10_features.csv
python gene_feature_pipeline.py --chrom chr11 --output chr11_features.csv
python gene_feature_pipeline.py --chrom chr12 --output chr12_features.csv
```

Each chromosome takes ~10–20 minutes (API rate limits). Or run all at once:

```bash
python run_all_chroms.py
```

---

### 3. Build the training dataset

```bash
python build_training_data.py
```

Joins all `chr*_features.csv` files to the VIS data. Outputs `training_data.csv` (~54 MB, ~392k rows).

RepeatMasker data is fetched from UCSC and cached in `data/rmsk_chrN.csv` — first run takes ~30 min, subsequent runs are instant.

**Output columns:**

| Column | Description |
|---|---|
| `label` | `log1p(max raw read count)` — training target, range ~1.4–12 |
| `dist_to_nearest_tss` | Distance (bp) from integration site to nearest gene's TSS |
| `in_gene_body` | 1 if site falls inside a gene, 0 otherwise |
| `dist_to_nearest_cancer_tss` | Distance to nearest cancer driver gene TSS |
| `gene_length` | Length of nearest gene (bp) |
| `in_repeat` | 1 if site is inside a repeat element |
| `cpg_island_count` | CpG islands overlapping nearest gene |
| `cpg_nearest_dist` | Distance to nearest CpG island (bp) |
| `cpg_mean_obs_exp` | Mean observed/expected CpG ratio |
| `encode_ccre_count` | ENCODE candidate cis-regulatory elements overlapping gene |
| `encode_promoter_count` | Promoter-like cCREs |
| `encode_enhancer_count` | Enhancer-like cCREs |
| `encode_ctcf_count` | CTCF insulator cCREs |
| `ot_cancer_hallmark_count` | Cancer hallmarks from Open Targets |
| `ot_disease_count` | Disease associations from Open Targets |
| `ot_is_cancer_driver` | 1 if gene is a known cancer driver |
| `n_samples_detected` | How many sequencing libraries the clone appeared in |
| `growth_rate` | Linear slope of read counts over time (reads/week) |
| `abundance_fold_change` | log1p(last timepoint) − log1p(first timepoint) |

---

### 4. Run the model

**Baseline (chr13 only, XGBoost):**

```bash
python xgboost_pilot/run_pilot.py
```

**Main model (chr13–16, LightGBM):**

```bash
python xgboost_pilot/run_chr13_16.py
```

---

## Training on different chromosomes

Open `xgboost_pilot/run_chr13_16.py` and change the three lines near the top:

```python
TRAIN_CHROMS = ["chr13", "chr14"]   # chromosomes the model learns from
VAL_CHROM   = "chr15"               # used for early stopping
TEST_CHROM  = "chr16"               # held out — never seen during training
```

Examples:

```python
# Train on more data, test on chr1
TRAIN_CHROMS = ["chr13", "chr14", "chr15", "chr16", "chr17"]
VAL_CHROM   = "chr18"
TEST_CHROM  = "chr1"

# Full genome except chr1 (requires chr10/11/12 feature files)
TRAIN_CHROMS = [f"chr{i}" for i in range(2, 23)] + ["chrX", "chrY"]
VAL_CHROM   = "chr22"
TEST_CHROM  = "chr1"
```

Then re-run:

```bash
python xgboost_pilot/run_chr13_16.py
```

The chromosome-stratified split means the test chromosome is completely unseen during training — this is a realistic measure of how the model generalises across the genome.

---

## Current results

| Setup | Train R2 | Val R2 | Test R2 |
|---|---|---|---|
| XGBoost, chr13 only | 0.09 | 0.003 | -0.02 |
| LightGBM, chr13+14 → chr16 | 0.35 | 0.29 | 0.31 |

Top predictive features (by gain): `n_samples_detected` > `growth_rate` > `abundance_fold_change` > `dist_to_nearest_tss`

The temporal features (how many times a clone was observed, whether it's growing) dominate genomic features — consistent with the biology.

---

## Notes

- **chr10, chr11, chr12** are missing from the feature tables and are excluded from `training_data.csv`. Add them with Step 2 above.
- `data/` and `training_data.csv` are in `.gitignore` — they are not in the repo and must be generated locally.
- All API calls use public endpoints, no credentials required.
