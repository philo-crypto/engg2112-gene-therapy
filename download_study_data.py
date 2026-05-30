#!/usr/bin/env python3
"""
Download Yan et al. (2023) SCID-X1 integration site data from Zenodo
and normalise it to a canonical CSV for downstream ML pipeline use.

Output: data/vis_raw.csv
Columns: chrom, site_pos, patient_id, timepoint, clonal_abundance
"""

import io, os, zipfile, logging, pathlib
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ZENODO_URL = (
    "https://zenodo.org/api/records/8147763/files/"
    "Supplemental_Data1.zip/content"
)
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vis_raw.csv")
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "Supplemental_Data1")
# The ZIP nests an extra folder inside, so actual files live here:
INNER_DIR  = os.path.join(EXTRACT_DIR, "Supplemental_Data1")

# Column name variants → canonical name
CHROM_ALIASES      = {"seqnames", "chr", "chromosome", "chrom"}
POS_ALIASES        = {"start", "pos", "position", "site", "site_pos", "chromstart"}
PATIENT_ALIASES    = {"patient", "patient_id", "id", "subject", "samplename", "sample"}
TIMEPOINT_ALIASES  = {"time", "timepoint", "tp", "month", "timepoints", "day"}
ABUNDANCE_ALIASES  = {"abundance", "reads", "count", "nreads", "soniclength",
                      "clonal_abundance", "readcount", "read_count", "sonicabundance",
                      "freq", "frequency", "estabund", "estimatedabundance"}
VALID_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def download_and_extract() -> list[pathlib.Path]:
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    log.info(f"Downloading Yan et al. 2023 data from Zenodo...")
    r = requests.get(ZENODO_URL, timeout=180)
    r.raise_for_status()
    log.info(f"  Downloaded {len(r.content) / 1024:.0f} KB")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    zf.extractall(EXTRACT_DIR)

    # ZIP nests an extra Supplemental_Data1 folder inside — find all .txt files
    root = pathlib.Path(INNER_DIR) if pathlib.Path(INNER_DIR).exists() else pathlib.Path(EXTRACT_DIR)
    extracted = [f for f in root.iterdir()
                 if f.is_file() and not f.name.startswith(".")]
    log.info(f"  Found {len(extracted)} data files in {root}/")
    for f in extracted:
        log.info(f"    {f.name}")
    return extracted


def _parse_vis_id(vis_series: pd.Series) -> pd.DataFrame:
    """
    Parse VIS identifiers of the form  chr1:263474-263475,-
    Returns DataFrame with columns: chrom, site_pos (0-based start).
    """
    # Strip surrounding quotes from R-style row names
    clean = vis_series.str.strip('"').str.strip("'")
    # Split on ':' → chrom | rest
    chrom = clean.str.split(":").str[0]
    rest  = clean.str.split(":").str[1]          # e.g. "263474-263475,-"
    # Start position is before the '-'
    site_pos = rest.str.split("-").str[0]
    return pd.DataFrame({"chrom": chrom, "site_pos": pd.to_numeric(site_pos, errors="coerce")})


def _parse_sample_col(col: str) -> tuple[str, str]:
    """
    Parse a sample column name like 'P5_CD3_12wks' or 'p1:bm_cd3_16wks'
    into (patient_id, timepoint).
    """
    parts = col.replace(":", "_").split("_")
    patient  = parts[0].upper() if parts else col
    timepoint = parts[-1] if len(parts) >= 2 else "unknown"
    return patient, timepoint


def normalise_vis(files: list[pathlib.Path]) -> pd.DataFrame:
    """
    Handle two VIS file formats found in Yan et al. 2023:

    Format A — P-files (SCID-X1, P1-P10):
      Row index = VIS id like "chr1:263474-263475,-"  (R row names, quoted)
      Columns   = sample abundances like P5_CD3_12wks
      → read with index_col=0, melt to long, parse row index for position

    Format B — T-files (CAR-T, T1-T7):
      Column 'vis' = VIS id like chr1:650300-650301,+
      Columns 'CD4', 'CD8' = per-lineage abundances
      → melt on vis column
    """
    frames = []
    for path in files:
        stem = path.stem  # e.g. P5_vis_and_samples
        log.info(f"  Processing {path.name}...")
        try:
            # Try reading with row index (P-file format)
            df = pd.read_csv(path, sep="\t", index_col=0)
        except Exception as e:
            log.warning(f"    Could not read {path.name}: {e} — skipping")
            continue

        cols_lower = [c.lower() for c in df.columns]

        # Format B: has a 'vis' column in the index or columns
        if df.index.name and df.index.name.lower() == "vis":
            df = df.reset_index().rename(columns={df.index.name: "vis"})
            vis_col = "vis"
        elif "vis" in cols_lower:
            df = df.reset_index(drop=True)
            vis_col = df.columns[cols_lower.index("vis")]
        else:
            vis_col = None

        if vis_col:
            # Format B: T-files
            sample_cols = [c for c in df.columns if c != vis_col]
            long = df.melt(id_vars=[vis_col], value_vars=sample_cols,
                           var_name="sample", value_name="clonal_abundance")
            coords = _parse_vis_id(long[vis_col])
            long["chrom"]      = coords["chrom"]
            long["site_pos"]   = coords["site_pos"]
            long["patient_id"] = stem.split("_")[0].upper()
            long["timepoint"]  = "unknown"
            long["sample_id"]  = long["patient_id"] + "_" + long["sample"]
            out = long[["chrom", "site_pos", "sample_id", "patient_id", "timepoint",
                        "clonal_abundance"]].copy()
        else:
            # Format A: P-files — row index is the VIS id, columns are sample abundances
            coords = _parse_vis_id(pd.Series(df.index, name="vis"))
            sample_cols = df.columns.tolist()
            long = df.copy()
            long["chrom"]    = coords["chrom"].values
            long["site_pos"] = coords["site_pos"].values
            long = long.melt(id_vars=["chrom", "site_pos"], value_vars=sample_cols,
                             var_name="sample", value_name="clonal_abundance")
            parsed = long["sample"].apply(_parse_sample_col)
            long["patient_id"] = parsed.apply(lambda x: x[0])
            long["timepoint"]  = parsed.apply(lambda x: x[1])
            long["sample_id"]  = long["sample"]   # full original col e.g. P5_CD3_12wks
            out = long[["chrom", "site_pos", "sample_id", "patient_id", "timepoint",
                        "clonal_abundance"]].copy()

        # Normalise chrom format and filter
        out["chrom"] = out["chrom"].apply(
            lambda c: c if str(c).startswith("chr") else f"chr{c}"
        )
        out = out[out["chrom"].isin(VALID_CHROMS)]
        out = out.dropna(subset=["site_pos", "clonal_abundance"])
        out["site_pos"] = out["site_pos"].astype(int)
        # Drop zero-abundance rows to keep data manageable
        out = out[out["clonal_abundance"] > 0]

        log.info(f"    {len(out):,} non-zero site-sample rows")
        if not out.empty:
            frames.append(out)

    if not frames:
        raise RuntimeError("No usable VIS data found. Check the log above.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Combined VIS data: {len(combined):,} rows, "
             f"{combined['chrom'].nunique()} chromosomes, "
             f"{combined['patient_id'].nunique()} patients")
    return combined


def main():
    if os.path.exists(OUTPUT_CSV):
        log.info(f"{OUTPUT_CSV} already exists — skipping download. Delete it to re-run.")
        df = pd.read_csv(OUTPUT_CSV)
        log.info(f"Existing file: {df.shape[0]:,} rows × {df.shape[1]} cols")
        log.info(f"Columns: {df.columns.tolist()}")
        log.info(f"Label range: {df['clonal_abundance'].min():.2f} – {df['clonal_abundance'].max():.2f}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Skip download if already extracted
    inner = pathlib.Path(INNER_DIR)
    if inner.exists() and any(inner.iterdir()):
        log.info(f"Already extracted — using files in {INNER_DIR}")
        files = [f for f in inner.iterdir() if f.is_file() and not f.name.startswith(".")]
    else:
        files = download_and_extract()

    log.info("Normalising VIS data...")
    vis_df = normalise_vis(files)

    vis_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(vis_df):,} rows → {OUTPUT_CSV}")
    log.info(f"Columns: {vis_df.columns.tolist()}")
    log.info(f"Unique samples (library groups): {vis_df['sample_id'].nunique()}")
    log.info(f"Abundance range: {vis_df['clonal_abundance'].min():.0f} – {vis_df['clonal_abundance'].max():.0f} (raw reads)")


if __name__ == "__main__":
    main()
