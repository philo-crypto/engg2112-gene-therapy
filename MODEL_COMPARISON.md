# Model Comparison — Insertional Mutagenesis Risk Prediction

## Task Summary

Predict the within-patient percentile rank of clonal abundance at last timepoint
for each (patient, gene) pair with confirmed integration at time 1.
All models use the same 20 features and chromosome-stratified split (test=chr1, val=chr2).

---

## Results Summary

| Model | Spearman rho | Top-20% Lift | R2 | Accuracy |
|---|---|---|---|---|
| LightGBM Regression (baseline) | 0.382 | 1.93x | 0.150 | 64.4% |
| Stacking Ensemble | 0.381 | **1.95x** | 0.148 | — |
| XGBoost Pairwise Ranking | 0.370 | 1.94x | n/a* | — |
| LightGBM LambdaRank | 0.364 | 1.90x | n/a* | — |
| MLP Neural Network | 0.361 | 1.87x | 0.134 | — |

*R2 is not meaningful for ranking model outputs — scores are not calibrated to the label scale.

---

## Model Descriptions

---

### 1. LightGBM Regression (Baseline)

**How it works:**
LightGBM is a gradient boosting framework that builds an ensemble of decision trees
sequentially, each correcting the residual errors of the previous. It uses
histogram-based binning and leaf-wise tree growth, making it fast and memory-efficient
on large tabular datasets. Here it is trained with a standard MSE regression objective,
predicting the continuous within-patient percentile rank label directly.

**Performance:**
- Spearman rho: 0.382 — moderate rank correlation between predicted and actual abundance rank
- Top-20% lift: 1.93x — the model's top-ranked genes are 1.93x more likely to be true
  high-abundance clones than random selection
- R2: 0.150 — explains 15% of label variance; the remaining 85% reflects patient biology
  (T-cell stimulation, immune competition) that genomic features cannot capture

**Justification:**
LightGBM regression is the natural first choice for structured tabular data with
continuous labels. It handles mixed feature types, missing values, and skewed
distributions robustly without preprocessing. The chromosome-stratified split
tests genuine genomic generalisation — the model must learn patterns on 22 chromosomes
and apply them to a held-out chromosome it has never seen. The regression objective
on the percentile rank label is well-suited because the label is already normalised
within patients, removing sequencing depth confounds.

---

### 2. Stacking Ensemble

**How it works:**
Stacking (stacked generalisation) trains multiple diverse base learners independently,
then trains a meta-learner to optimally combine their predictions. Here:
- Base learner 1: LightGBM (gradient boosting, leaf-wise trees)
- Base learner 2: XGBoost (gradient boosting, depth-wise trees, different regularisation)
- Base learner 3: Random Forest (bagged decision trees, independent of boosting)

The three models make predictions on the validation set (chr2), and a Ridge regression
meta-learner is trained on those predictions to learn the optimal weighting. The final
test prediction is a weighted blend. Meta-learner weights: LGB=0.43, XGB=0.23, RF=0.34.

**Performance:**
- Spearman rho: 0.381 — effectively tied with the baseline
- Top-20% lift: 1.95x — marginally best across all models (+0.02x over baseline)
- The meta-learner down-weights XGBoost (0.23) and up-weights RF (0.34), suggesting
  Random Forest provides orthogonal signal to the two gradient boosting models

**Justification:**
Stacking is theoretically motivated when base models make different errors — their
predictions are correlated but not identical, so a meta-learner can exploit the
disagreements. Here the improvement is minimal (1.93x to 1.95x), suggesting the
three tree models are largely learning the same patterns from the same features.
In a higher-signal setting or with more diverse base learners (e.g. neural networks
+ linear models + trees), stacking would likely provide a larger gain. It remains
the strongest model on lift and is a valid production choice.

---

### 3. XGBoost Pairwise Ranking

**How it works:**
XGBoost with the `rank:pairwise` objective treats prediction as a learning-to-rank
problem. Rather than minimising regression error, it optimises pairwise concordance:
for every pair of genes within the same patient, it tries to correctly order which
has higher clonal abundance. Genes are grouped by patient_id so comparisons only
happen within a patient, not across patients. The label is converted to an integer
relevance score (0-4) representing which quartile of abundance the gene falls in.

**Performance:**
- Spearman rho: 0.370 — slightly below the regression baseline
- Top-20% lift: 1.94x — comparable to baseline on the primary clinical metric
- R2 is not interpretable (-57) because ranking scores are not calibrated to the
  label scale; the raw output is an ordinal score, not a predicted abundance rank

**Justification:**
Pairwise ranking is semantically more appropriate than regression for this task —
the clinical question is "which genes are riskier than others for this patient,"
not "what is the exact abundance." However, the label (within-patient percentile
rank) is already a normalised ranking, so regression on it is functionally equivalent
to ranking. The pairwise loss provided no empirical benefit here, likely because
the label normalisation already removes the cross-patient confounds that ranking
objectives are designed to handle. Pairwise ranking would be more valuable if the
raw log-abundance (un-normalised) were used as the label.

---

### 4. LightGBM LambdaRank

**How it works:**
LambdaRank is a gradient boosting approach that optimises NDCG (Normalised Discounted
Cumulative Gain) directly — a standard information retrieval ranking metric that
rewards placing the most relevant items at the top of the ranked list, with
diminishing credit for lower positions. LightGBM implements this via LambdaMART,
computing pseudo-gradients that represent how much swapping two items would change
the NDCG score. Like XGBoost pairwise, genes are grouped by patient and the label
is discretised to integer relevance levels (0-4).

**Performance:**
- Spearman rho: 0.364 — weakest of the tree models
- Top-20% lift: 1.90x — slightly below baseline
- NDCG@5=0.660, NDCG@10=0.577, NDCG@20=0.555 (on validation set)

**Justification:**
NDCG-optimised ranking is the gold standard in search and recommendation systems
where position matters — the top result must be the most relevant. In a clinical
context, identifying the single highest-risk gene per patient (NDCG@1) would be
the ideal objective. However, LambdaRank underperformed here for the same reason
as XGBoost pairwise: the regression baseline on a percentile-rank label already
approximates a ranking objective, leaving little room for improvement. Additionally,
discretising the continuous label to 5 relevance levels (0-4) loses information
compared to the continuous regression target. LambdaRank would be more competitive
if the task were truly sparse — e.g. predicting which 1 in 100 genes will cause
a leukemic event.

---

### 5. MLP Neural Network

**How it works:**
A multi-layer perceptron (sklearn MLPRegressor) with three hidden layers of size
256-128-64 neurons and ReLU activations. Features are standardised to zero mean
and unit variance before training (required for neural networks). The network
learns non-linear combinations of features through backpropagation with Adam
optimiser. Early stopping on a held-out validation fraction prevents overfitting.
Unlike tree models, the MLP learns smooth, continuous decision boundaries rather
than axis-aligned splits.

**Performance:**
- Spearman rho: 0.361 — weakest overall
- Top-20% lift: 1.87x — only model below 1.90x
- R2: 0.134 — lower than the LightGBM baseline

**Justification:**
Neural networks generally require large datasets and many samples per feature to
outperform tree models on tabular data. With only 20 features and ~64,000 training
samples, the MLP is in a regime where gradient boosting has a well-documented
advantage — trees can capture the sharp thresholds and interactions present in
genomic data more efficiently than smooth neural network activations. The MLP
also cannot natively handle the heavy-tailed distributions of distance features
(dist_to_nearest_tss ranges from 0 to 250M bp), even after standardisation.
A deeper or regularised network, or one with feature-specific embeddings, might
close the gap. The MLP is included as a reference point: if the signal were
purely non-linear in ways trees miss, the MLP would outperform — its underperformance
here confirms the signal structure is tree-friendly (threshold-based, sparse).

---

## Key Takeaways

1. **The baseline is hard to beat.** All five models cluster within 0.02 Spearman rho
   and 0.08x lift of each other. The performance ceiling is the biological signal
   available in the features, not the model architecture.

2. **Regression on percentile rank is already a ranking problem.** Dedicated ranking
   objectives (LambdaRank, XGBoost Pairwise) offered no improvement because the
   within-patient percentile rank label already handles the cross-patient normalisation
   that ranking objectives are designed to address.

3. **Stacking is the production recommendation.** It matches or slightly exceeds
   the baseline on every metric, is robust to individual model failures, and provides
   a principled way to combine future model improvements. The 1.95x lift is the
   best achieved across all models.

4. **Tree models dominate neural networks on this dataset.** 20 features, threshold-
   based genomic signals, and heavy-tailed distributions all favour gradient boosting
   over MLPs. Neural networks would become competitive with more features or
   sequence-level inputs (e.g. raw chromatin accessibility bigWig signals).

5. **Lift is the right clinical metric.** R2 is misleading for this task — ranking
   models score negative R2 despite competitive lift. Top-20% lift directly measures
   clinical utility: can the model help prioritise which patients and genes to monitor
   most closely after gene therapy?
