#!/usr/bin/env python3
"""
Run gene feature pipeline for all chromosomes except chr1 (already done).
Downloads ENCODE BED once and reuses it for all chromosomes.
"""

import sys, time, gzip, logging
import requests
import pandas as pd

sys.path.insert(0, ".")
from gene_feature_pipeline import (
    _get, get_genes_on_chromosome, get_cpg_islands, annotate_cpg, annotate_encode,
    ENCODE_BASE, RATE_PAUSE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CHROMS = [f"chr{i}" for i in range(2, 23)] + ["chrX", "chrY"]


def download_encode_bed_all() -> pd.DataFrame:
    """Download full ENCODE cCRE BED once, return DataFrame with all chromosomes."""
    log.info("Fetching ENCODE annotation metadata...")
    search = _get(
        f"{ENCODE_BASE}/search/",
        params={
            "type":            "Annotation",
            "annotation_type": "candidate Cis-Regulatory Elements",
            "assembly":        "GRCh38",
            "status":          "released",
            "limit":           10,
            "format":          "json",
        },
        headers={"Accept": "application/json"},
    )
    graph = search.get("@graph", [])
    if not graph:
        raise RuntimeError("No ENCODE cCRE annotation found.")

    annotation_id = graph[0]["@id"]
    annotation = _get(
        f"{ENCODE_BASE}{annotation_id}",
        params={"format": "json"},
        headers={"Accept": "application/json"},
    )

    download_url = None
    for f in annotation.get("files", []):
        file_meta = f if isinstance(f, dict) else _get(
            f"{ENCODE_BASE}{f}", params={"format": "json"}, headers={"Accept": "application/json"}
        )
        if (
            file_meta.get("file_format") == "bed"
            and "GRCh38" in str(file_meta.get("assembly", ""))
            and file_meta.get("status") == "released"
            and file_meta.get("href")
        ):
            download_url = f"https://www.encodeproject.org{file_meta['href']}"
            break

    if not download_url:
        raise RuntimeError("Could not locate a released BED file in ENCODE annotation.")

    log.info(f"Downloading ENCODE BED from {download_url}...")
    r = requests.get(download_url, timeout=600)
    r.raise_for_status()
    raw = gzip.decompress(r.content) if download_url.endswith(".gz") else r.content
    lines = raw.decode(errors="replace").splitlines()

    rows = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        rows.append({
            "chrom":        parts[0],
            "encode_start": int(parts[1]),
            "encode_end":   int(parts[2]),
            "encode_name":  parts[3] if len(parts) > 3 else "",
            "encode_type":  parts[9] if len(parts) > 9 else parts[3] if len(parts) > 3 else "",
        })

    df = pd.DataFrame(rows)
    log.info(f"Total cCREs loaded: {len(df):,}")
    return df


def run_chrom(chrom: str, encode_all: pd.DataFrame) -> None:
    out = f"{chrom.replace('chr', 'chr')}_features.csv"
    log.info(f"\n=== {chrom} ===")

    genes = get_genes_on_chromosome(chrom)
    if genes.empty:
        log.warning(f"No genes found for {chrom}, skipping.")
        return

    cpg = get_cpg_islands(chrom)
    genes = annotate_cpg(genes, cpg)

    encode_chrom = encode_all[encode_all["chrom"] == chrom].drop(columns=["chrom"]).reset_index(drop=True)
    log.info(f"  {len(encode_chrom):,} cCREs on {chrom}")
    genes = annotate_encode(genes, encode_chrom)

    genes.to_csv(out, index=False)
    log.info(f"  Saved {len(genes)} rows -> {out}")


if __name__ == "__main__":
    encode_all = download_encode_bed_all()

    for chrom in CHROMS:
        try:
            run_chrom(chrom, encode_all)
        except Exception as e:
            log.error(f"Failed {chrom}: {e}")
        time.sleep(RATE_PAUSE)

    log.info("\nAll chromosomes done.")
