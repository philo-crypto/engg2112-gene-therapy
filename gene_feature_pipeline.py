#!/usr/bin/env python3
"""
Gene feature pipeline
Pulls all protein-coding genes on one chromosome from Ensembl, then enriches
each gene with COSMIC mutation burden, UCSC CpG islands, and ENCODE cCREs.

Requirements:
    pip install requests pandas

COSMIC credentials:
    Register at https://cancer.sanger.ac.uk/cosmic/register
    Then set COSMIC_EMAIL and COSMIC_PASSWORD env vars (or pass them in).
"""

import time
import logging
import requests
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── API base URLs ─────────────────────────────────────────────────────────────
ENSEMBL_BASE = "https://rest.ensembl.org"
UCSC_BASE    = "https://api.genome.ucsc.edu"
ENCODE_BASE  = "https://www.encodeproject.org"

RATE_PAUSE = 0.25  # seconds between requests to avoid rate-limiting


# ── Shared HTTP helper ────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, headers: dict = None, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 429:
                wait = int(e.response.headers.get("Retry-After", 10))
                log.warning(f"Rate limited — waiting {wait}s")
                time.sleep(wait)
            elif code == 404:
                return {}
            elif attempt == retries - 1:
                raise
            else:
                time.sleep(2 ** attempt)
    return {}


# ── 1. ENSEMBL: gene locations ────────────────────────────────────────────────

def _chrom_length(chrom: str) -> int:
    data = _get(
        f"{ENSEMBL_BASE}/info/assembly/homo_sapiens",
        params={"content-type": "application/json"},
    )
    clean = chrom.replace("chr", "")
    for region in data.get("top_level_region", []):
        if region["name"] == clean:
            return region["length"]
    raise ValueError(f"Chromosome '{chrom}' not found in Ensembl assembly info")


def get_genes_on_chromosome(chrom: str, chunk_size: int = 5_000_000) -> pd.DataFrame:
    """
    Return a DataFrame of all protein-coding genes on a chromosome.
    Queries Ensembl in 5 Mb chunks to stay within API limits.
    """
    clean = chrom.replace("chr", "")
    total = _chrom_length(clean)
    log.info(f"chr{clean} length: {total:,} bp — fetching genes in {chunk_size//1_000_000} Mb chunks")

    records = []
    for start in range(1, total + 1, chunk_size):
        end = min(start + chunk_size - 1, total)
        region = f"{clean}:{start}-{end}"
        log.info(f"  Ensembl region {region}")
        genes = _get(
            f"{ENSEMBL_BASE}/overlap/region/homo_sapiens/{region}",
            params={"feature": "gene", "content-type": "application/json"},
        )
        if not isinstance(genes, list):
            continue
        for g in genes:
            if g.get("biotype") == "protein_coding":
                records.append({
                    "gene_id":   g["id"],
                    "gene_name": g.get("external_name", ""),
                    "chrom":     f"chr{clean}",
                    "start":     g["start"],
                    "end":       g["end"],
                    "strand":    g["strand"],
                })
        time.sleep(RATE_PAUSE)

    df = (
        pd.DataFrame(records)
        .drop_duplicates("gene_id")
        .sort_values("start")
        .reset_index(drop=True)
    )
    log.info(f"Found {len(df)} protein-coding genes on chr{clean}")
    return df


# ── 2. Open Targets: cancer gene annotations (replaces COSMIC) ───────────────
# COSMIC's REST API was retired in their Next.js migration; Open Targets provides
# equivalent cancer driver / hallmark data via a free GraphQL API using Ensembl IDs.

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_OT_QUERY = """
query GeneInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    hallmarks {
      cancerHallmarks { label impact }
    }
    associatedDiseases(page: {index: 0, size: 1}) { count }
  }
}
"""


def get_opentargets_gene_data(ensembl_id: str) -> dict:
    empty = {"ot_cancer_hallmark_count": None, "ot_disease_count": None, "ot_is_cancer_driver": None}
    if not ensembl_id:
        return empty
    try:
        r = requests.post(
            OPENTARGETS_URL,
            json={"query": _OT_QUERY, "variables": {"ensemblId": ensembl_id}},
            timeout=30,
        )
        r.raise_for_status()
        target = (r.json().get("data") or {}).get("target")
        if not target:
            return empty
        hallmarks  = (target.get("hallmarks") or {}).get("cancerHallmarks") or []
        unique_hallmarks = len({h["label"] for h in hallmarks})
        disease_count    = (target.get("associatedDiseases") or {}).get("count", 0)
        return {
            "ot_cancer_hallmark_count": unique_hallmarks,
            "ot_disease_count":         disease_count,
            "ot_is_cancer_driver":      int(unique_hallmarks > 0),
        }
    except Exception as e:
        log.debug(f"Open Targets miss for {ensembl_id}: {e}")
        return empty


def enrich_with_opentargets(genes_df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Fetching Open Targets data for {len(genes_df)} genes...")
    rows = []
    for _, row in genes_df.iterrows():
        rows.append(get_opentargets_gene_data(row["gene_id"]))
        time.sleep(RATE_PAUSE)
    return pd.concat([genes_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ── 3. UCSC: CpG islands ──────────────────────────────────────────────────────

def get_cpg_islands(chrom: str) -> pd.DataFrame:
    """Download every CpG island on a chromosome from the UCSC REST API."""
    ucsc_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    log.info(f"Fetching CpG islands for {ucsc_chrom} from UCSC...")
    data = _get(
        f"{UCSC_BASE}/getData/track",
        params={"genome": "hg38", "track": "cpgIslandExt", "chrom": ucsc_chrom},
    )
    islands = data.get("cpgIslandExt", [])
    if not islands:
        log.warning("No CpG islands returned from UCSC.")
        return pd.DataFrame(columns=["cpg_start", "cpg_end", "cpg_obs_exp", "cpg_per_cpg", "cpg_per_gc"])

    df = pd.DataFrame(islands)[["chromStart", "chromEnd", "obsExp", "perCpg", "perGc"]]
    df.columns = ["cpg_start", "cpg_end", "cpg_obs_exp", "cpg_per_cpg", "cpg_per_gc"]
    log.info(f"  {len(df)} CpG islands on {ucsc_chrom}")
    return df


def annotate_cpg(genes_df: pd.DataFrame, cpg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per gene:
      cpg_island_count  — how many CpG islands overlap the gene body
      cpg_nearest_dist  — bp to nearest island (0 when overlapping)
      cpg_mean_obs_exp  — mean obs/exp CpG ratio of overlapping islands
    """
    results = []
    for _, g in genes_df.iterrows():
        gs, ge = g["start"], g["end"]
        overlap = cpg_df[(cpg_df["cpg_start"] < ge) & (cpg_df["cpg_end"] > gs)]
        if len(overlap):
            results.append({
                "cpg_island_count": len(overlap),
                "cpg_nearest_dist": 0,
                "cpg_mean_obs_exp": round(overlap["cpg_obs_exp"].mean(), 4),
            })
        else:
            dist = min(
                (cpg_df["cpg_start"] - ge).abs().min(),
                (cpg_df["cpg_end"]   - gs).abs().min(),
            ) if len(cpg_df) else None
            results.append({
                "cpg_island_count": 0,
                "cpg_nearest_dist": int(dist) if dist is not None else None,
                "cpg_mean_obs_exp": None,
            })

    return pd.concat([genes_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ── 4. ENCODE: candidate cis-Regulatory Elements (cCREs) ─────────────────────

def get_encode_ccres(chrom: str) -> pd.DataFrame:
    """
    Download the GRCh38 ENCODE cCRE annotation BED file via the ENCODE portal
    search API, filter to the requested chromosome, and return a DataFrame.

    cCRE types in the BED name field:
      PLS   — promoter-like signature
      pELS  — proximal enhancer-like signature
      dELS  — distal enhancer-like signature
      CTCF-only — insulator
      DNase-H3K4me3 — DNase + H3K4me3 signal
    """
    ucsc_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    log.info(f"Searching ENCODE portal for GRCh38 cCRE annotation...")

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
        log.warning("No ENCODE cCRE annotation found — check ENCODE portal manually.")
        return pd.DataFrame()

    # Pick the first released annotation and find its BED file
    annotation_id = graph[0]["@id"]
    log.info(f"  Using annotation {annotation_id}")
    annotation = _get(
        f"{ENCODE_BASE}{annotation_id}",
        params={"format": "json"},
        headers={"Accept": "application/json"},
    )

    download_url = None
    for f in annotation.get("files", []):
        # The ENCODE API sometimes embeds full file objects inline, sometimes returns path strings
        if isinstance(f, str):
            file_meta = _get(
                f"{ENCODE_BASE}{f}",
                params={"format": "json"},
                headers={"Accept": "application/json"},
            )
        else:
            file_meta = f
        if (
            file_meta.get("file_format") == "bed"
            and "GRCh38" in str(file_meta.get("assembly", ""))
            and file_meta.get("status") == "released"
            and file_meta.get("href")
        ):
            download_url = f"https://www.encodeproject.org{file_meta['href']}"
            break

    if not download_url:
        log.warning("Could not locate a released BED file in this ENCODE annotation.")
        return pd.DataFrame()

    log.info(f"  Downloading cCRE BED from {download_url} (filtering to {ucsc_chrom})")
    import gzip, io
    r = requests.get(download_url, timeout=300)
    r.raise_for_status()

    # Decompress if gzipped
    raw = gzip.decompress(r.content) if download_url.endswith(".gz") else r.content
    lines = raw.decode(errors="replace").splitlines()

    rows = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] != ucsc_chrom:
            continue
        # BED6+: chrom, start, end, name, score, strand …
        rows.append({
            "encode_start": int(parts[1]),
            "encode_end":   int(parts[2]),
            "encode_name":  parts[3] if len(parts) > 3 else "",
            "encode_type":  parts[9] if len(parts) > 9 else parts[3] if len(parts) > 3 else "",
        })

    df = pd.DataFrame(rows)
    log.info(f"  {len(df)} cCREs on {ucsc_chrom}")
    return df


def annotate_encode(genes_df: pd.DataFrame, encode_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per gene:
      encode_ccre_count      — total overlapping cCREs
      encode_promoter_count  — PLS (promoter-like) overlaps
      encode_enhancer_count  — pELS + dELS (enhancer-like) overlaps
      encode_ctcf_count      — CTCF-only (insulator) overlaps
    """
    if encode_df.empty:
        for col in ["encode_ccre_count", "encode_promoter_count",
                    "encode_enhancer_count", "encode_ctcf_count"]:
            genes_df[col] = None
        return genes_df

    results = []
    for _, g in genes_df.iterrows():
        gs, ge = g["start"], g["end"]
        ov = encode_df[(encode_df["encode_start"] < ge) & (encode_df["encode_end"] > gs)]
        t = ov["encode_type"].str.upper()
        results.append({
            "encode_ccre_count":     len(ov),
            "encode_promoter_count": t.str.contains("PLS",  na=False).sum(),
            "encode_enhancer_count": t.str.contains("ELS",  na=False).sum(),
            "encode_ctcf_count":     t.str.contains("CTCF", na=False).sum(),
        })

    return pd.concat([genes_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ── 5. Pipeline entry point ───────────────────────────────────────────────────

def build_feature_table(
    chrom: str = "chr1",
    output_csv: str = "gene_features.csv",
) -> pd.DataFrame:
    """
    Full pipeline for one chromosome.
    Returns enriched DataFrame and writes it to output_csv.

    Args:
        chrom      chromosome in 'chr1' or '1' format
        output_csv path for the output CSV
    """
    log.info(f"=== Building feature table for {chrom} ===")

    genes   = get_genes_on_chromosome(chrom)
    genes   = enrich_with_opentargets(genes)
    cpg     = get_cpg_islands(chrom)
    genes   = annotate_cpg(genes, cpg)
    encode  = get_encode_ccres(chrom)
    genes   = annotate_encode(genes, encode)

    genes.to_csv(output_csv, index=False)
    log.info(f"Saved {len(genes)} rows → {output_csv}")
    return genes


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a gene feature table for one chromosome.")
    parser.add_argument("--chrom",  default="chr1",             help="Chromosome (default: chr1)")
    parser.add_argument("--output", default="gene_features.csv", help="Output CSV path")
    args = parser.parse_args()

    df = build_feature_table(
        chrom=args.chrom,
        output_csv=args.output,
    )

    print("\n-- Feature table preview --")
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{df.dtypes.to_string()}")
