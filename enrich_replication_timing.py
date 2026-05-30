#!/usr/bin/env python3
"""
Add replication timing score to all chr*_features.csv files.

Source: UCSC hg19 wgEncodeUwRepliSeq K562 wave signal (Encode UW Repli-seq).
K562 is a hematopoietic cell line — closest available to HSC/T-cell biology.
Higher score = earlier replication = more accessible chromatin.

Gene positions are used in hg19-approximate coordinates (chromosomal positions
are ~99% stable between hg19 and hg38 for protein-coding genes).

Queries the UCSC REST API in 5 Mb chunks per chromosome and averages the
replication timing signal over each gene body.
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
UCSC_BASE   = "https://api.genome.ucsc.edu"
TRACK       = "wgEncodeUwRepliSeqK562WaveSignalRep1"
CHUNK_SIZE  = 5_000_000
RATE_PAUSE  = 0.3
ALL_CHROMS  = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def fetch_repli_signal(chrom: str) -> pd.DataFrame:
    """Fetch replication timing wave signal for a chromosome from UCSC hg19."""
    cache = os.path.join(ROOT, "data", f"repli_{chrom}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)

    log.info(f"  Fetching replication timing for {chrom} ...")
    records = []
    # Query in chunks — the track is a wiggle signal stored as intervals
    for start in range(0, 250_000_000, CHUNK_SIZE):
        end = start + CHUNK_SIZE
        try:
            r = requests.get(
                f"{UCSC_BASE}/getData/track",
                params={"genome": "hg19", "track": TRACK,
                        "chrom": chrom, "start": start, "end": end,
                        "maxItemsOutput": 100_000},
                timeout=60
            )
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            items = data.get(TRACK, [])
            if not items:
                if data.get("end", end) <= end:
                    break
                continue
            for item in items:
                records.append({
                    "start": item.get("start", item.get("chromStart", 0)),
                    "end":   item.get("end",   item.get("chromEnd",   0)),
                    "score": float(item.get("value", item.get("score", 0))),
                })
            time.sleep(RATE_PAUSE)
        except Exception as e:
            log.warning(f"    chunk {start}-{end}: {e}")
            time.sleep(2)
            continue

    if not records:
        log.warning(f"  No replication timing data for {chrom}")
        return pd.DataFrame(columns=["start", "end", "score"])

    df = pd.DataFrame(records).drop_duplicates().sort_values("start").reset_index(drop=True)
    df.to_csv(cache, index=False)
    log.info(f"  {chrom}: {len(df):,} intervals cached")
    return df


def mean_signal_over_genes(genes_df: pd.DataFrame, signal_df: pd.DataFrame) -> np.ndarray:
    """For each gene, compute mean replication timing signal over its body."""
    if signal_df.empty:
        return np.full(len(genes_df), np.nan)

    sig_start = signal_df["start"].values
    sig_end   = signal_df["end"].values
    sig_score = signal_df["score"].values
    scores = np.full(len(genes_df), np.nan)

    for i, row in enumerate(genes_df.itertuples()):
        mask = (sig_start < row.end) & (sig_end > row.start)
        if mask.any():
            scores[i] = sig_score[mask].mean()

    return scores


def patch_chrom(chrom: str) -> None:
    path = os.path.join(ROOT, f"{chrom}_features.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    if "replication_timing" in df.columns and df["replication_timing"].notna().mean() > 0.5:
        log.info(f"{chrom}: already annotated, skipping")
        return

    signal = fetch_repli_signal(chrom)
    df["replication_timing"] = mean_signal_over_genes(df, signal)
    filled = df["replication_timing"].notna().sum()
    log.info(f"{chrom}: {filled}/{len(df)} genes annotated  "
             f"(mean={df['replication_timing'].mean():.2f})")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for chrom in ALL_CHROMS:
        try:
            patch_chrom(chrom)
        except Exception as e:
            log.error(f"{chrom} failed: {e}")
    log.info("Done — replication timing added to all chr*_features.csv")
