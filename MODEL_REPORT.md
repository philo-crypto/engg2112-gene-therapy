# Insertional Mutagenesis Risk Model — Report

## Model Overview

**Architecture:** LightGBM regression (chromosome-stratified split)  
**Task:** Predict within-patient percentile rank of clonal abundance at last timepoint, given only genomic features of the nearest gene  
**Split:** Test = chr1, Val = chr2, Train = all other chromosomes  
**Dataset:** 76,402 (patient, gene) pairs across 17 SCID-X1 patients (Yan et al. 2023)

---

## Performance

| Split | R² | RMSE | Spearman ρ | Accuracy (median) | Top-20% Lift |
|---|---|---|---|---|---|
| Train (chr3–22, X, Y) | 0.194 | 0.256 | 0.438 | 65.9% | 2.06x |
| Val (chr2) | 0.121 | 0.263 | 0.349 | 62.2% | 1.76x |
| **Test (chr1)** | **0.150** | **0.262** | **0.382** | **64.4%** | **1.93x** |

**Key metric — Top-20% lift:** The model identifies the top 20% highest-risk (patient, gene) pairs with 1.93x better precision than random selection. A random selector would capture 20% of true high-risk pairs; the model captures 38.6%.

---

## Features (20 total, ranked by importance gain)

| Feature | Gain | Description |
|---|---|---|
| `ext_log_unique_sites` | 7,633 | log1p(unique integration sites near this gene across 59 external patients) |
| `ext_log_n_patients` | 1,536 | log1p(external patients with any integration near this gene) |
| `replication_timing` | 448 | Mean K562 replication timing signal over gene body (hg19 UW Repli-seq) |
| `dist_to_nearest_cancer_tss` | 121 | Distance (bp) to nearest cancer driver gene TSS |
| `dist_to_nearest_tss` | 99 | Distance (bp) to nearest gene TSS |
| `tcell_h3k27ac_count` | 51 | H3K27ac ChIP-seq peaks overlapping gene body (CD4+ T-cells, ENCODE) |
| `ext_n_datasets` | 29 | Number of independent external datasets with integration near this gene |
| `ot_disease_count` | 28 | Open Targets: number of associated diseases |
| `encode_ccre_count` | 25 | ENCODE candidate cis-regulatory elements overlapping gene |
| `cpg_island_count` | 22 | CpG islands overlapping gene body |
| `gene_length` | 20 | Gene body length (bp) — longer genes present larger integration targets |
| `cpg_mean_obs_exp` | 18 | Mean observed/expected CpG ratio (promoter methylation proxy) |
| `encode_ctcf_count` | 15 | CTCF insulator binding sites near gene |
| `encode_enhancer_count` | 14 | ENCODE enhancer-like elements near gene |
| `cpg_nearest_dist` | 12 | Distance to nearest CpG island (0 if overlapping) |
| `encode_promoter_count` | 9 | ENCODE promoter-like elements near gene |
| `ot_cancer_hallmark_count` | 5 | Open Targets: cancer hallmarks associated with gene |
| `in_repeat` | 3 | Whether integration site falls within a RepeatMasker repeat element |
| `ot_is_cancer_driver` | 0 | Binary: gene is a known cancer driver (Open Targets) |
| `ext_max_abundance` | 0 | Max clonal abundance observed near gene in external data |

---

## External Data Sources

The two highest-importance features come from pooling integration site data across independent trials:

| Dataset | Disease | Vector | Patients | Observations |
|---|---|---|---|---|
| Calabria et al. 2024 (Nature) | MLD / WAS / β-Thalassemia | Lentiviral | 53 | 10.9M |
| Bushman lab 2020 (Blood) | WAS / SCD / β-Thalassemia | Lentiviral | 6 | 801K |

**Total: 59 patients, 11.8M integration observations → 14,994 genes annotated**

---

## Key Takeaways

**1. Integration hotspots dominate the signal.**  
`ext_log_unique_sites` alone explains the majority of predictive gain. Genes that lentiviral vectors consistently target across 59 independent patients from three diseases and two labs are in genuinely open chromatin — and are more likely to be hit (and thus expand) in the Yan SCID-X1 cohort.

**2. Replication timing is the strongest pure-genomic feature.**  
Early-replicating regions (high K562 replication timing score) are preferentially targeted by lentiviral vectors, consistent with the known biology that integration correlates with active DNA synthesis. This ranks 3rd overall despite being a single scalar per gene.

**3. T-cell chromatin accessibility adds independent signal.**  
H3K27ac peaks in CD4+ T-cells (ENCODE ENCFF112ULV) contribute modest but non-zero gain, confirming that target cell-type-specific chromatin state matters beyond generic cCRE annotations.

**4. Cancer driver annotations (OT) have near-zero predictive gain.**  
`ot_is_cancer_driver` and `ot_cancer_hallmark_count` rank last. This is expected for a lentiviral safe trial — the Yan cohort shows no insertional oncogenesis. These features would likely gain importance in gamma-retroviral datasets where clonal dominance near LMO2/EVI1 actually occurred.

**5. The performance ceiling is biological, not technical.**  
All four model architectures tested (LightGBM, XGBoost, Random Forest, Ridge) converge to ~1.9x lift. The remaining ~85% of label variance is driven by T-cell biology (antigen stimulation, clonal competition, patient conditioning) that no static genomic feature can capture. The model is learning *where vectors insert*, not *which clones survive* — which is the correct and achievable prediction task.

**6. Label design matters.**  
Within-patient percentile rank normalisation (rather than raw log abundance) was critical to remove sequencing depth confounding between patients, enabling the model to learn cross-patient genomic signal rather than fitting per-patient technical differences.
