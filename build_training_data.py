#!/usr/bin/env python3
"""
Build the XGBoost training dataset by joining Yan et al. integration site data
with the genomic gene feature tables.

Inputs:
  data/vis_raw.csv         — from download_study_data.py
  chr*_features.csv        — from gene_feature_pipeline.py / enrich pipelines

Output:
  training_data.csv        — one row per unique integration site, 21 columns

Target variable: label = log(1 + max_clonal_abundance)
"""

import os, glob, logging, time, re
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")
from gene_feature_pipeline import _get, UCSC_BASE, RATE_PAUSE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

GENE_COLS_REQUIRED = [
    "gene_id", "gene_name", "chrom", "start", "end", "strand",
    "cpg_island_count", "cpg_nearest_dist", "cpg_mean_obs_exp",
    "encode_ccre_count", "encode_promoter_count", "encode_enhancer_count",
    "encode_ctcf_count", "ot_cancer_hallmark_count", "ot_disease_count",
    "ot_is_cancer_driver", "tcell_h3k27ac_count", "replication_timing",
]
INHERITED_COLS = [
    "cpg_island_count", "cpg_nearest_dist", "cpg_mean_obs_exp",
    "encode_ccre_count", "encode_promoter_count", "encode_enhancer_count",
    "encode_ctcf_count", "ot_cancer_hallmark_count", "ot_disease_count",
    "ot_is_cancer_driver", "tcell_h3k27ac_count", "replication_timing",
]

COSMIC_COLS   = ["cosmic_tier", "cosmic_total_mutations", "cosmic_cancer_types"]
TEMPORAL_COLS = ["n_samples_detected", "growth_rate", "abundance_fold_change", "n_patients_with_site"]
RMSK_CACHE_DIR = "data"


def _parse_timepoint_weeks(tp: str) -> float:
    """Convert a timepoint string like '12wks', '6mo', '24' to numeric weeks."""
    if not tp or str(tp).lower() in ("unknown", "nan", ""):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    if not m:
        return np.nan
    val = float(m.group(1))
    if "mo" in str(tp).lower():
        val *= 4.33
    return val


# ── 1. Load gene catalogue ────────────────────────────────────────────────────

def load_gene_catalogue(feature_dir: str = ".") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(feature_dir, "chr*_features.csv")))
    if not files:
        raise FileNotFoundError(f"No chr*_features.csv files found in {feature_dir}")
    log.info(f"Loading {len(files)} chromosome feature files...")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        # Drop COSMIC cols if present (not in the required feature set)
        df = df.drop(columns=[c for c in COSMIC_COLS if c in df.columns])
        # Normalise schema — fill missing OT cols with NaN
        df = df.reindex(columns=GENE_COLS_REQUIRED)
        frames.append(df)

    genes = pd.concat(frames, ignore_index=True)
    genes["tss_pos"]    = np.where(genes["strand"] == 1, genes["start"], genes["end"])
    genes["gene_length"] = genes["end"] - genes["start"]
    # Treat NaN ot_is_cancer_driver as 0 for filtering purposes
    genes["ot_is_cancer_driver"] = genes["ot_is_cancer_driver"].fillna(0).astype(int)

    genes = genes.sort_values(["chrom", "tss_pos"]).reset_index(drop=True)
    log.info(f"Gene catalogue: {len(genes):,} genes across {genes['chrom'].nunique()} chromosomes")
    return genes


# ── 2. Load VIS data and compute labels ──────────────────────────────────────

def load_vis_data(vis_csv: str = "data/vis_raw.csv") -> pd.DataFrame:
    if not os.path.exists(vis_csv):
        raise FileNotFoundError(f"{vis_csv} not found — run download_study_data.py first")
    df = pd.read_csv(vis_csv)
    required = {"chrom", "site_pos", "sample_id", "patient_id", "clonal_abundance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vis_raw.csv is missing columns: {missing}. Re-run download_study_data.py.")
    log.info(f"VIS data: {len(df):,} rows, {df['chrom'].nunique()} chromosomes, "
             f"{df['patient_id'].nunique()} patients, {df['sample_id'].nunique()} samples")
    return df


def compute_labels(vis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per patient-site pair, compute:
      label                — log1p(max raw read count), range ~[1.4, 12]
      max_raw_abundance    — raw read count maximum across all samples
      max_rel_abundance    — within-sample relative abundance maximum (diagnostic)
      n_samples_detected   — how many sequencing libraries the site appeared in
      growth_rate          — linear slope of reads over time (reads/week); 0 if
                             fewer than 2 timepoints have known numeric values
      abundance_fold_change — log1p(last_abundance) - log1p(first_abundance)
                              along the time axis; 0 if < 2 known timepoints
    """
    vis_df = vis_df.copy()

    # Within-sample relative abundance
    sample_totals = vis_df.groupby("sample_id")["clonal_abundance"].transform("sum")
    vis_df["rel_abundance"] = vis_df["clonal_abundance"] / sample_totals

    # Numeric timepoints for slope calculation
    vis_df["tp_weeks"] = vis_df["timepoint"].apply(_parse_timepoint_weeks)

    # Simple aggregations — vectorised
    simple = (
        vis_df
        .groupby(["patient_id", "chrom", "site_pos"], sort=False)
        .agg(
            max_raw_abundance=("clonal_abundance", "max"),
            max_rel_abundance=("rel_abundance",    "max"),
            n_samples_detected=("sample_id",       "count"),
        )
        .reset_index()
    )

    # Growth rate and fold-change — require per-group time series
    def _trajectory(grp):
        raw = grp["clonal_abundance"].values
        tw  = grp["tp_weeks"].values
        valid = ~np.isnan(tw)
        if valid.sum() >= 2:
            t_v, a_v = tw[valid], raw[valid]
            order = np.argsort(t_v)
            slope       = float(np.polyfit(t_v, a_v, 1)[0])
            fold_change = float(np.log1p(a_v[order[-1]]) - np.log1p(a_v[order[0]]))
        else:
            slope = 0.0
            fold_change = 0.0
        return pd.Series({"growth_rate": slope, "abundance_fold_change": fold_change})

    traj = (
        vis_df
        .groupby(["patient_id", "chrom", "site_pos"], sort=False)
        .apply(_trajectory)
        .reset_index()
    )

    agg = simple.merge(traj, on=["patient_id", "chrom", "site_pos"])

    # Label: log1p(max raw read count) as per proposal
    agg["label"] = np.log1p(agg["max_raw_abundance"])

    # Site recurrence: how many unique patients share this integration site
    site_patients = (
        vis_df.groupby(["chrom", "site_pos"])["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients_with_site"})
    )
    agg = agg.merge(site_patients, on=["chrom", "site_pos"])

    log.info(f"Unique patient-site pairs: {len(agg):,}")
    log.info(f"Label (log1p raw) range: {agg['label'].min():.3f} - {agg['label'].max():.3f}")
    log.info(f"Label mean: {agg['label'].mean():.3f}  std: {agg['label'].std():.3f}")
    log.info(f"growth_rate > 0: {(agg['growth_rate'] > 0).sum():,} sites")
    return agg


# ── 3. RepeatMasker from UCSC ─────────────────────────────────────────────────

def get_repeat_masker(chrom: str, chrom_length: int, chunk_size: int = 10_000_000) -> pd.DataFrame:
    cache_path = os.path.join(RMSK_CACHE_DIR, f"rmsk_{chrom}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        log.info(f"  rmsk {chrom}: loaded {len(df):,} records from cache")
        return df

    log.info(f"  Fetching RepeatMasker for {chrom} from UCSC ({chrom_length/1e6:.0f} Mb)...")
    rows = []
    for start in range(0, chrom_length, chunk_size):
        end = min(start + chunk_size, chrom_length)
        data = _get(
            f"{UCSC_BASE}/getData/track",
            params={
                "genome": "hg38",
                "track": "rmsk",
                "chrom": chrom,
                "start": start,
                "end": end,
                "maxItemsOutput": 1_000_000,
            },
        )
        records = data.get("rmsk", [])
        for r in records:
            rows.append({"rmsk_start": int(r["genoStart"]), "rmsk_end": int(r["genoEnd"])})
        time.sleep(RATE_PAUSE)

    df = pd.DataFrame(rows).drop_duplicates().sort_values("rmsk_start").reset_index(drop=True)
    os.makedirs(RMSK_CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    log.info(f"  rmsk {chrom}: {len(df):,} records")
    return df


# ── 4. Vectorised spatial join functions ──────────────────────────────────────

def find_nearest_gene(
    site_arr: np.ndarray,
    tss_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nearest_gene_index, dist_to_tss) for each site. O(S log G)."""
    idx = np.searchsorted(tss_arr, site_arr)
    left  = np.clip(idx - 1, 0, len(tss_arr) - 1)
    right = np.clip(idx,     0, len(tss_arr) - 1)
    dist_left  = np.abs(site_arr - tss_arr[left])
    dist_right = np.abs(site_arr - tss_arr[right])
    nearest_idx = np.where(dist_left <= dist_right, left, right)
    dist_to_tss = np.abs(site_arr - tss_arr[nearest_idx])
    return nearest_idx, dist_to_tss


def find_nearest_cancer_gene(
    site_arr: np.ndarray,
    genes_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nearest_cancer_gene_index_into_genes_df, dist) for each site."""
    cancer = genes_df[genes_df["ot_is_cancer_driver"] == 1].copy()
    n = len(site_arr)
    if cancer.empty:
        return np.full(n, -1, dtype=int), np.full(n, np.nan)

    cancer_tss = cancer["tss_pos"].values
    cancer_idx_in_cancer, dist = find_nearest_gene(site_arr, cancer_tss)
    # Map back to indices in the original genes_df
    cancer_global_idx = cancer.index.values[cancer_idx_in_cancer]
    return cancer_global_idx, dist


def _in_intervals(site_arr: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Vectorised: return 0/1 array indicating whether each site falls within
    any interval [starts[i], ends[i]]. Requires starts to be sorted.
    """
    result = np.zeros(len(site_arr), dtype=int)
    if len(starts) == 0:
        return result
    idx = np.searchsorted(starts, site_arr, side="right") - 1
    valid = idx >= 0
    result[valid] = (ends[idx[valid]] >= site_arr[valid]).astype(int)
    return result


def in_gene_body(site_arr: np.ndarray, genes_df: pd.DataFrame) -> np.ndarray:
    gdf = genes_df.sort_values("start")
    return _in_intervals(site_arr, gdf["start"].values, gdf["end"].values)


def in_repeat(site_arr: np.ndarray, rmsk_df: pd.DataFrame) -> np.ndarray:
    if rmsk_df.empty:
        return np.zeros(len(site_arr), dtype=int)
    return _in_intervals(
        site_arr,
        rmsk_df["rmsk_start"].values,
        rmsk_df["rmsk_end"].values,
    )


# ── 5. Per-chromosome feature computation ────────────────────────────────────

def compute_features(
    sites_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    rmsk_df: pd.DataFrame,
) -> pd.DataFrame:
    sites_df = sites_df.sort_values("site_pos").reset_index(drop=True)
    genes_df = genes_df.sort_values("tss_pos").reset_index(drop=True)

    site_arr = sites_df["site_pos"].values
    tss_arr  = genes_df["tss_pos"].values

    # Nearest gene
    near_idx, dist_tss = find_nearest_gene(site_arr, tss_arr)

    # Nearest cancer driver gene
    cancer_idx, dist_cancer = find_nearest_cancer_gene(site_arr, genes_df)
    has_cancer = cancer_idx >= 0
    nearest_cancer_name = np.where(
        has_cancer,
        genes_df["gene_name"].values[np.clip(cancer_idx, 0, len(genes_df)-1)],
        None,
    )

    # In gene body
    in_body = in_gene_body(site_arr, genes_df)

    # In repeat
    in_rpt = in_repeat(site_arr, rmsk_df)

    out = sites_df[["patient_id", "chrom", "site_pos", "label",
                    "max_raw_abundance", "max_rel_abundance"] + TEMPORAL_COLS].copy()
    out["dist_to_nearest_tss"]      = dist_tss
    out["nearest_gene_name"]        = genes_df["gene_name"].values[near_idx]
    out["in_gene_body"]             = in_body
    out["dist_to_nearest_cancer_tss"] = dist_cancer
    out["nearest_cancer_gene_name"] = nearest_cancer_name
    out["gene_length"]              = genes_df["gene_length"].values[near_idx]
    out["in_repeat"]                = in_rpt

    for col in INHERITED_COLS:
        out[col] = genes_df[col].values[near_idx]

    return out


# ── 6. Main orchestration ─────────────────────────────────────────────────────

def main(
    feature_dir: str = ".",
    vis_csv:     str = "data/vis_raw.csv",
    output_csv:  str = "training_data.csv",
):
    genes_all = load_gene_catalogue(feature_dir)
    vis_df    = load_vis_data(vis_csv)
    sites_all = compute_labels(vis_df)

    results = []
    for chrom in CHROMS:
        sites_c = sites_all[sites_all["chrom"] == chrom]
        genes_c = genes_all[genes_all["chrom"] == chrom]

        if sites_c.empty:
            log.info(f"{chrom}: no integration sites — skipping")
            continue
        if genes_c.empty:
            log.warning(f"{chrom}: no genes in feature table — skipping {len(sites_c)} sites")
            continue

        chrom_len = int(genes_c["end"].max()) + 5_000_000
        log.info(f"\n{chrom}: {len(sites_c):,} sites × {len(genes_c):,} genes")

        rmsk_c = get_repeat_masker(chrom, chrom_len)
        feat_c = compute_features(sites_c, genes_c, rmsk_c)
        results.append(feat_c)
        log.info(f"  Done — {len(feat_c):,} rows")

    if not results:
        raise RuntimeError("No results produced — check VIS data and feature CSV files")

    training = pd.concat(results, ignore_index=True)

    # Label-encode patient_id so XGBoost/LightGBM can use it as a numeric feature
    training["patient_id_enc"] = pd.Categorical(training["patient_id"]).codes

    training.to_csv(output_csv, index=False)

    log.info(f"\n{'='*50}")
    log.info(f"Saved {len(training):,} rows × {len(training.columns)} cols → {output_csv}")
    log.info(f"Columns: {training.columns.tolist()}")
    log.info(f"Label:   min={training['label'].min():.3f}  "
             f"max={training['label'].max():.3f}  "
             f"mean={training['label'].mean():.3f}")
    log.info(f"In gene body: {training['in_gene_body'].sum():,} / {len(training):,}")
    log.info(f"In repeat:    {training['in_repeat'].sum():,} / {len(training):,}")
    log.info(f"Near cancer driver: "
             f"{(training['dist_to_nearest_cancer_tss'] < 50_000).sum():,} sites within 50 kb of a cancer gene")


if __name__ == "__main__":
    main()
