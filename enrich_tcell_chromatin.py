#!/usr/bin/env python3
"""
Add T-cell H3K27ac ChIP-seq peak counts to all chr*_features.csv files.

Source: ENCODE ENCFF112ULV — replicated H3K27ac peaks, CD4+ memory T cells, GRCh38.
Adds: tcell_h3k27ac_count — number of H3K27ac peaks overlapping each gene body.

Also fetches ATAC-seq peaks from ENCODE for T helper 17 cells (ENCSR803FKU).
Adds: tcell_atac_count — number of ATAC-seq peaks overlapping each gene body.
"""

import os, sys, gzip, io, logging, requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ENCODE_H3K27AC_URL = "https://www.encodeproject.org/files/ENCFF112ULV/@@download/ENCFF112ULV.bed.gz"
ENCODE_BASE = "https://www.encodeproject.org"

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def download_bed(url: str, cache_path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(cache_path):
        log.info(f"Downloading {label} ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
        log.info(f"  saved {len(r.content)/1e6:.1f} MB -> {cache_path}")
    else:
        log.info(f"{label} already cached")

    with gzip.open(cache_path, "rt") as f:
        rows = []
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rows.append({"chrom": parts[0], "start": int(parts[1]), "end": int(parts[2])})
    df = pd.DataFrame(rows)
    df = df[df["chrom"].str.match(r"^chr(\d+|X|Y)$")]
    log.info(f"  {label}: {len(df):,} peaks loaded")
    return df


def get_atac_url() -> str:
    """Find the replicated peaks BED file for ENCSR803FKU (Th17 ATAC-seq)."""
    cache = os.path.join(DATA_DIR, "encsr803fku_meta.json")
    if os.path.exists(cache):
        import json
        with open(cache) as f:
            return json.load(f)["url"]

    log.info("Fetching ENCODE ATAC-seq metadata for ENCSR803FKU ...")
    r = requests.get(f"{ENCODE_BASE}/experiments/ENCSR803FKU/",
                     params={"format": "json"}, headers={"Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    data = r.json()

    for f in data.get("files", []):
        file_meta = f if isinstance(f, dict) else requests.get(
            f"{ENCODE_BASE}{f}", params={"format": "json"},
            headers={"Accept": "application/json"}, timeout=30
        ).json()
        if (file_meta.get("file_format") == "bed"
                and file_meta.get("output_type", "").lower() in ("replicated peaks", "peaks")
                and "GRCh38" in str(file_meta.get("assembly", ""))
                and file_meta.get("status") == "released"
                and file_meta.get("href")):
            url = f"{ENCODE_BASE}{file_meta['href']}"
            import json
            with open(cache, "w") as fc:
                json.dump({"url": url}, fc)
            log.info(f"  ATAC peaks URL: {url}")
            return url

    log.warning("Could not find ATAC peaks BED for ENCSR803FKU")
    return None


def count_peaks_per_gene(genes_df: pd.DataFrame, peaks_df: pd.DataFrame, col_name: str) -> pd.Series:
    """Vectorised overlap count: how many peaks overlap each gene body."""
    counts = np.zeros(len(genes_df), dtype=int)
    for chrom, g_grp in genes_df.groupby("chrom"):
        p = peaks_df[peaks_df["chrom"] == chrom]
        if p.empty:
            continue
        p_start = p["start"].values
        p_end   = p["end"].values
        for i, (_, g) in enumerate(g_grp.iterrows()):
            gs, ge = g["start"], g["end"]
            mask = (p_start < ge) & (p_end > gs)
            counts[genes_df.index.get_loc(g.name)] = mask.sum()
    return pd.Series(counts, index=genes_df.index, name=col_name)


def patch_chrom(chrom: str, h3k27ac_df: pd.DataFrame, atac_df: pd.DataFrame) -> None:
    path = os.path.join(ROOT, f"{chrom}_features.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    if "tcell_h3k27ac_count" in df.columns and "tcell_atac_count" in df.columns:
        log.info(f"{chrom}: already annotated, skipping")
        return

    log.info(f"{chrom}: annotating {len(df)} genes ...")
    h3k = h3k27ac_df[h3k27ac_df["chrom"] == chrom]
    atac = atac_df[atac_df["chrom"] == chrom] if atac_df is not None else pd.DataFrame()

    # Vectorised overlap using searchsorted
    def overlap_counts(genes, peaks):
        if peaks.empty:
            return np.zeros(len(genes), dtype=int)
        ps = peaks["start"].values
        pe = peaks["end"].values
        counts = np.zeros(len(genes), dtype=int)
        for i, row in enumerate(genes.itertuples()):
            mask = (ps < row.end) & (pe > row.start)
            counts[i] = mask.sum()
        return counts

    df["tcell_h3k27ac_count"] = overlap_counts(df, h3k)
    df["tcell_atac_count"]    = overlap_counts(df, atac) if not atac.empty else 0

    df.to_csv(path, index=False)
    h3k_hits = (df["tcell_h3k27ac_count"] > 0).sum()
    log.info(f"  {chrom}: {h3k_hits}/{len(df)} genes with H3K27ac peaks")


if __name__ == "__main__":
    h3k27ac = download_bed(
        ENCODE_H3K27AC_URL,
        os.path.join(DATA_DIR, "encff112ulv_h3k27ac.bed.gz"),
        "H3K27ac CD4 T-cell peaks (ENCFF112ULV)"
    )

    atac_url = get_atac_url()
    atac = download_bed(
        atac_url,
        os.path.join(DATA_DIR, "encsr803fku_atac.bed.gz"),
        "ATAC-seq Th17 peaks (ENCSR803FKU)"
    ) if atac_url else None

    for chrom in ALL_CHROMS:
        try:
            patch_chrom(chrom, h3k27ac, atac)
        except Exception as e:
            log.error(f"{chrom} failed: {e}")

    log.info("Done — T-cell chromatin features added to all chr*_features.csv")
