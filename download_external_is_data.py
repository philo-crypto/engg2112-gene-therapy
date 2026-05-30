#!/usr/bin/env python3
"""
Download and process external integration site datasets, then compute per-gene
features from them to enrich patient_gene_features_v2.csv.

Sources:
  1. Calabria et al. 2024 (Nature) — MLD/WAS/β-Thal lentiviral, 42 patients
     GitHub: calabrialab/Code_HSPCdynamics  (tsv.gz per patient)
  2. Bushman lab Blood 2020 — WAS/SCD/β-Thal lentiviral, 6 patients
     GitHub: BushmanLab/HSC_diversity  (intSites.MergedSamples.csv.gz)

Output: data/external_is_features.csv
  One row per nearest_gene_name with:
    ext_n_patients       — unique patients with integration near this gene
    ext_n_datasets       — independent datasets contributing observations
    ext_n_unique_sites   — unique integration sites near gene
    ext_max_abundance    — max fragment/sonic abundance at any site near gene
"""

import os, io, sys, zipfile, logging
import requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "external_is_features.csv")
os.makedirs(DATA_DIR, exist_ok=True)

GENE_COLS_NEEDED = ["chrom", "nearest_gene_name", "site_pos"]


# ── helpers ───────────────────────────────────────────────────────────────────

def download_bytes(url: str, desc: str) -> bytes:
    log.info(f"Downloading {desc} ...")
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    chunks = []
    for chunk in r.iter_content(1024 * 1024):
        chunks.append(chunk)
    data = b"".join(chunks)
    log.info(f"  {len(data)/1e6:.1f} MB received")
    return data


def load_gene_site_map() -> pd.DataFrame:
    """Load (chrom, site_pos, nearest_gene_name) from training_data.csv."""
    path = os.path.join(ROOT, "training_data.csv")
    df = pd.read_csv(path, usecols=GENE_COLS_NEEDED, low_memory=False)
    df = df.drop_duplicates(subset=["chrom", "site_pos"])
    log.info(f"Gene-site map: {len(df):,} unique sites across {df['nearest_gene_name'].nunique():,} genes")
    return df


def map_to_genes(sites_df: pd.DataFrame, gene_map: pd.DataFrame) -> pd.DataFrame:
    """
    Nearest-gene lookup via sorted searchsorted per chromosome.
    sites_df must have columns: chrom, site_pos
    Returns sites_df with nearest_gene_name added.
    """
    results = []
    for chrom, grp in sites_df.groupby("chrom"):
        gm = gene_map[gene_map["chrom"] == chrom].sort_values("site_pos")
        if gm.empty:
            continue
        tss = gm["site_pos"].values
        genes = gm["nearest_gene_name"].values
        pos = grp["site_pos"].values
        idx = np.searchsorted(tss, pos)
        left  = np.clip(idx - 1, 0, len(tss) - 1)
        right = np.clip(idx,     0, len(tss) - 1)
        nearest = np.where(
            np.abs(pos - tss[left]) <= np.abs(pos - tss[right]), left, right
        )
        grp = grp.copy()
        grp["nearest_gene_name"] = genes[nearest]
        results.append(grp)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ── Dataset 1: Calabria 2024 ──────────────────────────────────────────────────

def load_calabria() -> pd.DataFrame:
    """Download repo ZIP, parse all per-patient tsv.gz files in data/."""
    cache = os.path.join(DATA_DIR, "calabria_repo.zip")
    if not os.path.exists(cache):
        raw = download_bytes(
            "https://github.com/calabrialab/Code_HSPCdynamics/archive/refs/heads/main.zip",
            "Calabria 2024 repo ZIP"
        )
        with open(cache, "wb") as f:
            f.write(raw)
    else:
        log.info("Calabria ZIP already cached")

    records = []
    with zipfile.ZipFile(cache) as zf:
        tsv_files = [n for n in zf.namelist()
                     if n.endswith(".tsv.gz") and "/data/" in n
                     and any(d in n for d in ["/MLD/", "/WAS/", "/BTHAL/"])]
        log.info(f"Calabria: {len(tsv_files)} tsv.gz files")

        for fname in tsv_files:
            # Extract dataset (MLD/WAS/BTHAL) and patient from path
            parts = fname.split("/")
            dataset = next((p for p in parts if p in ("MLD", "WAS", "BTHAL")), "UNK")
            # Patient ID from filename e.g. Pt01.20200101_MLD_SubjID_...
            pt_part = parts[-1].split(".")[0]  # e.g. "Pt01"
            patient_id = f"calabria_{dataset}_{pt_part}"

            # Only use fragmentEstimate files (skip seqCount duplicates)
            if "seqCount" in fname:
                continue

            try:
                with zf.open(fname) as f:
                    import gzip
                    with gzip.open(f) as gz:
                        df = pd.read_csv(gz, sep="\t", low_memory=False)
            except Exception as e:
                log.debug(f"  skip {fname}: {e}")
                continue

            # Normalise column names
            chr_col  = next((c for c in df.columns if c.lower() in ("chr", "chromosome", "seqnames")), None)
            pos_col  = next((c for c in df.columns if c.lower() in ("integration_locus", "position", "start", "pos")), None)
            abun_col = next((c for c in df.columns if c.lower() in ("fragmentestimate", "seqcount", "estabund", "abundance")), None)

            if not chr_col or not pos_col:
                log.debug(f"  skip {fname}: missing chr/pos columns {df.columns.tolist()[:8]}")
                continue

            sub = pd.DataFrame({
                "chrom":     df[chr_col].astype(str).str.strip(),
                "site_pos":  pd.to_numeric(df[pos_col], errors="coerce"),
                "abundance": pd.to_numeric(df[abun_col], errors="coerce") if abun_col else np.nan,
                "patient_id": patient_id,
                "dataset":   "calabria_2024",
            }).dropna(subset=["site_pos"])
            records.append(sub)

    if not records:
        log.warning("No Calabria records parsed")
        return pd.DataFrame()

    df = pd.concat(records, ignore_index=True)
    # Ensure chr prefix
    df["chrom"] = df["chrom"].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
    df = df[df["chrom"].str.match(r"^chr(\d+|X|Y)$")]
    df["site_pos"] = df["site_pos"].astype(int)
    log.info(f"Calabria: {len(df):,} site obs, {df['patient_id'].nunique()} patients")
    return df


# ── Dataset 2: Bushman Blood 2020 ─────────────────────────────────────────────

def load_bushman() -> pd.DataFrame:
    cache = os.path.join(DATA_DIR, "bushman_intSites.csv.gz")
    if not os.path.exists(cache):
        for branch in ("master", "main"):
            url = f"https://github.com/BushmanLab/HSC_diversity/raw/{branch}/data/intSites.MergedSamples.csv.gz"
            try:
                raw = download_bytes(url, f"Bushman 2020 (branch={branch})")
                with open(cache, "wb") as f:
                    f.write(raw)
                break
            except Exception as e:
                log.warning(f"  {branch} failed: {e}")
        else:
            log.warning("Bushman download failed on all branches — skipping")
            return pd.DataFrame()
    else:
        log.info("Bushman CSV already cached")

    df = pd.read_csv(cache, low_memory=False)
    log.info(f"Bushman raw: {len(df):,} rows, columns: {df.columns.tolist()[:12]}")

    # posid format: "chr1+100027349" or "chr1-100027349"
    if "posid" in df.columns:
        split = df["posid"].str.extract(r"^(chr[\dXY]+)[+-](\d+)$")
        df["chrom"]    = split[0]
        df["site_pos"] = pd.to_numeric(split[1], errors="coerce")
    elif "chr" in df.columns and "position" in df.columns:
        df["chrom"]    = df["chr"]
        df["site_pos"] = pd.to_numeric(df["position"], errors="coerce")

    abun_col = next((c for c in df.columns if c.lower() in ("estabund", "fragmentestimate", "abundance")), None)
    pat_col  = next((c for c in df.columns if c.lower() in ("patient", "subject", "patientid")), None)

    out = pd.DataFrame({
        "chrom":      df["chrom"],
        "site_pos":   df["site_pos"],
        "abundance":  pd.to_numeric(df[abun_col], errors="coerce") if abun_col else np.nan,
        "patient_id": df[pat_col].astype(str).apply(lambda x: f"bushman_{x}") if pat_col else "bushman_unknown",
        "dataset":    "bushman_2020",
    }).dropna(subset=["chrom", "site_pos"])

    out = out[out["chrom"].str.match(r"^chr(\d+|X|Y)$")]
    out["site_pos"] = out["site_pos"].astype(int)
    log.info(f"Bushman: {len(out):,} site obs, {out['patient_id'].nunique()} patients")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    gene_map = load_gene_site_map()

    log.info("\n=== Loading Calabria 2024 ===")
    cal = load_calabria()

    log.info("\n=== Loading Bushman 2020 ===")
    bush = load_bushman()

    all_sites = pd.concat([cal, bush], ignore_index=True)
    log.info(f"\nCombined: {len(all_sites):,} site obs, "
             f"{all_sites['patient_id'].nunique()} patients, "
             f"{all_sites['dataset'].nunique()} datasets")

    # Take max abundance per (patient, site) across timepoints
    per_patient_site = (
        all_sites.groupby(["dataset", "patient_id", "chrom", "site_pos"])["abundance"]
        .max().reset_index()
    )

    log.info("\n=== Mapping sites to nearest genes ===")
    mapped = map_to_genes(per_patient_site, gene_map)
    log.info(f"  Mapped: {len(mapped):,} / {len(per_patient_site):,} site-patient rows")

    log.info("\n=== Computing per-gene external features ===")
    gene_feats = (
        mapped.groupby("nearest_gene_name")
        .agg(
            ext_n_patients   =("patient_id",  "nunique"),
            ext_n_datasets   =("dataset",     "nunique"),
            ext_n_unique_sites=("site_pos",   "nunique"),
            ext_max_abundance=("abundance",   "max"),
        )
        .reset_index()
    )
    gene_feats["ext_max_abundance"] = gene_feats["ext_max_abundance"].fillna(0)

    log.info(f"  Genes with external IS data: {len(gene_feats):,}")
    log.info(f"  ext_n_patients range: {gene_feats['ext_n_patients'].min()} - {gene_feats['ext_n_patients'].max()}")
    log.info(f"  ext_n_datasets range: {gene_feats['ext_n_datasets'].min()} - {gene_feats['ext_n_datasets'].max()}")

    gene_feats.to_csv(OUT_PATH, index=False)
    log.info(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
