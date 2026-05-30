#!/usr/bin/env python3
"""
Patch Open Targets annotations into chr*_features.csv files missing OT data.

Chromosomes to patch: chr2, chr3-chr9, chr12, chr22, chrX, chrY
Estimated runtime: ~35 min (0.25 s per gene × ~8,300 genes)

Resumes from partial runs: skips genes already annotated (ot_is_cancer_driver not NaN).
"""

import sys, time, logging
import pandas as pd

sys.path.insert(0, ".")
from gene_feature_pipeline import get_opentargets_gene_data, RATE_PAUSE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MISSING_CHROMS = ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                  "chr12", "chr22", "chrX", "chrY"]
OT_COLS = ["ot_cancer_hallmark_count", "ot_disease_count", "ot_is_cancer_driver"]


def patch_chrom(chrom: str) -> None:
    path = f"{chrom}_features.csv"
    df = pd.read_csv(path)

    if "gene_id" not in df.columns:
        log.error(f"{chrom}: no gene_id column — cannot enrich. Skipping.")
        return

    # Ensure OT columns exist
    for col in OT_COLS:
        if col not in df.columns:
            df[col] = None

    # Find rows that still need annotation
    needs_annotation = df["ot_is_cancer_driver"].isna()
    n_todo = needs_annotation.sum()
    n_total = len(df)
    log.info(f"{chrom}: {n_total} genes, {n_todo} need OT annotation")

    if n_todo == 0:
        log.info(f"{chrom}: already fully annotated, skipping.")
        return

    completed = 0
    for idx in df.index[needs_annotation]:
        gene_id = df.at[idx, "gene_id"]
        result = get_opentargets_gene_data(str(gene_id) if pd.notna(gene_id) else "")
        for col in OT_COLS:
            df.at[idx, col] = result[col]
        completed += 1
        if completed % 100 == 0:
            log.info(f"  {chrom}: {completed}/{n_todo} done — saving checkpoint")
            df.to_csv(path, index=False)
        time.sleep(RATE_PAUSE)

    df.to_csv(path, index=False)
    driver_pct = df["ot_is_cancer_driver"].fillna(0).gt(0).mean() * 100
    log.info(f"{chrom}: done. Cancer drivers: {driver_pct:.1f}%")


if __name__ == "__main__":
    for chrom in MISSING_CHROMS:
        try:
            patch_chrom(chrom)
        except Exception as e:
            log.error(f"Failed {chrom}: {e}")
    log.info("\nAll chromosomes patched.")
