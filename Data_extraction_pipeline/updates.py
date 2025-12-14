"""
update_pipeline.py

Production-ready orchestration for: download PubMed updatefiles -> transform XML -> filter -> update baseline CSV.

Features:
- safe backup of baseline
- duplicate handling (drop duplicates by PubMedID, keep last)
- vectorized update logic (no iterrows)
- simple unit checks/assertions
- logging + progress messages
- configurable workers / threads
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Existing modules
import extract_baseline as extr
import transform_xml_to_csv as transf
import filter as filter

# ---------- Configuration ----------
NUM_THREADS = 10
NUM_WORKERS = os.cpu_count() or 1

# Paths to the two main datasets
BASELINE_FILE_NAME = Path("Result-v5/results_protein_matches.csv")
UPDATES_FILE_NAME = Path("Result-updates/results_protein_matches.csv")

# Global variables for the transform_xml_to_csv code
transf.OUTPUT_DIR = Path("Pubmed_abstracts_updates_csvs")
transf.INPUT_DIR = Path("Pubmed_Updates")
transf.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
transf.NUM_WORKERS = NUM_WORKERS

# Global variables for the extract_baseline code
extr.URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
extr.OUTPUT_DIR = Path("Pubmed_Updates")
extr.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for the filter code
filter.DATA_FOLDER = str(transf.OUTPUT_DIR)
filter.OUT_FOLDER = str(Path("Result-updates"))
Path(filter.OUT_FOLDER).mkdir(parents=True, exist_ok=True)

filter.MODEL_NAME = "en_ner_jnlpba_md"
filter.PROTEIN_LIST = "protein_synonyms.csv"
filter.SIM_THRESHOLD = 0.85
filter.MIN_TOKEN_OVERLAP = 0.7
filter.BATCH_SIZE = 500
filter.N_WORKERS = None

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("update_pipeline")


# ---------- Helpers ----------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with sane defaults for large files and give informative error."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def backup_file(path: Path, keep: int = 5) -> Path:
    """Make a timestamped backup of `path`. Return backup path."""
    if not path.exists():
        return None
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".backup.{stamp}")
    shutil.copy2(path, dest)
    log.info(f"Backup created: {dest}")
    # Optional: prune old backups (not implemented beyond 'keep' param)
    return dest


# ---------- Pipeline steps (light wrappers around your modules) ----------
def updates_extraction():
    """List and download update files in parallel using extract_baseline module."""
    log.info("Fetching list of update files from PubMed FTP...")
    files = extr.list_files(extr.URL)
    log.info(f"Found {len(files)} files to download.")

    if not files:
        log.info("No files to download.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(extr.download_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(files), desc="Download progress"):
            try:
                results.append(fut.result())
            except Exception as e:
                # extr.download_file should ideally return (filename, status) but guard anyway
                log.exception("Download task raised exception: %s", e)

    # Summarize if results are in the form [(name, status), ...]
    try:
        downloaded = sum(1 for _, status in results if status == "downloaded")
        skipped = sum(1 for _, status in results if status == "exists")
        failed = [f for f, status in results if isinstance(status, str) and status.startswith("error")]
        log.info("SUMMARY: downloaded=%d skipped=%d failed=%d", downloaded, skipped, len(failed))
    except Exception:
        log.debug("Could not compute summary of results (unexpected results format).")


def xml_to_csv():
    """Transform downloaded .xml.gz files into CSVs using transform_xml_to_csv.extract_pubmed_data"""
    files = sorted(transf.INPUT_DIR.glob("*.xml.gz"))
    if not files:
        log.info("No .xml.gz files found in %s", transf.INPUT_DIR)
        return

    log.info("Found %d XML files. Processing with %d workers", len(files), transf.NUM_WORKERS)

    futures = []
    with ProcessPoolExecutor(max_workers=transf.NUM_WORKERS) as executor:
        for xml_file in files:
            out_file = transf.OUTPUT_DIR / (xml_file.stem.replace(".xml", "") + ".csv")
            # Skip already processed files
            if out_file.exists():
                continue
            futures.append(executor.submit(transf.extract_pubmed_data, xml_file, out_file))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing XML files"):
            pass

    # Summary
    ok = 0
    failed = []
    for fut in futures:
        try:
            fname, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                failed.append((fname, status))
        except Exception as e:
            log.exception("Worker raised exception: %s", e)
            failed.append(("unknown", str(e)))

    log.info("Finished. Successfully processed: %d Failed: %d", ok, len(failed))
    if failed:
        for f, err in failed:
            log.warning("Failed: %s : %s", f, err)


def run_filter():
    """Run your filter module to produce Result-updates/results_protein_matches.csv"""
    log.info("Running filter.main() to generate updates file(s)...")
    filter.main()
    log.info("Filter step produced: %s", UPDATES_FILE_NAME)


# ---------- Core update logic ----------
def compare_and_update(baseline: pd.DataFrame, update: pd.DataFrame, *,
                       baseline_path: Path,
                       backup: bool = True) -> pd.DataFrame:
    """
    Vectorized update of baseline with update DataFrame.
    - baseline: existing baseline DataFrame (contains PubMedID, Matched_Proteins, Relevant_Sentences)
    - update: update DataFrame (contains PubMedID, Matched_Proteins, Relevant_Sentences)
    Returns the updated baseline DataFrame.
    """

    required_cols = {"PubMedID", "Matched_Proteins", "Relevant_Sentences"}
    for df_name, df in (("baseline", baseline), ("update", update)):
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"{df_name} is missing required columns: {required_cols - set(df.columns)}")

    # Save original counts for unit checks
    baseline_before = baseline.shape[0]
    unique_before = baseline['PubMedID'].nunique()

    # Merge to identify new vs existing
    merged = update.merge(
        baseline[['PubMedID', 'Matched_Proteins', 'Relevant_Sentences']],
        on='PubMedID', how='left', suffixes=('_new', '_old')
    )

    #### IDENTIFYING NEW ROWS ####
    new_mask = merged['Matched_Proteins_old'].isna()  # baseline missing -> new
    new_rows = merged.loc[new_mask, ['PubMedID', 'Matched_Proteins_new', 'Relevant_Sentences_new']].rename(
        columns={'Matched_Proteins_new': 'Matched_Proteins', 'Relevant_Sentences_new': 'Relevant_Sentences'}
    )

    #### IDENTIFYING REVISED ROWS ####
    # Use fillna('') to avoid NaN weirdness in comparisons
    matched_new = merged['Matched_Proteins_new'].fillna('').astype(str)
    matched_old = merged['Matched_Proteins_old'].fillna('').astype(str)
    sent_new = merged['Relevant_Sentences_new'].fillna('').astype(str)
    sent_old = merged['Relevant_Sentences_old'].fillna('').astype(str)

    diff_mask = (matched_new != matched_old) | (sent_new != sent_old)
    existing_mask = merged['Matched_Proteins_old'].notna()
    revised_mask = existing_mask & diff_mask

    revised_ids = merged.loc[revised_mask, 'PubMedID'].unique().tolist()
    new_ids = new_rows['PubMedID'].unique().tolist()

    log.info("Found %d new rows and %d revised rows in updates.", len(new_ids), len(revised_ids))

    #### ADDING MODIFICATIONS TO BASELINE ####
    # Apply revised updates to baseline using mapping (vectorized)
    if revised_ids:
        # create mapping from update PubMedID -> new values
        upd_map_mp = update.set_index('PubMedID')['Matched_Proteins'].astype(object).to_dict()
        upd_map_rs = update.set_index('PubMedID')['Relevant_Sentences'].astype(object).to_dict()

        # update baseline in-place (vectorized)
        mask_baseline_revised = baseline['PubMedID'].isin(revised_ids)
        baseline.loc[mask_baseline_revised, 'Matched_Proteins'] = baseline.loc[mask_baseline_revised, 'PubMedID'].map(upd_map_mp)
        baseline.loc[mask_baseline_revised, 'Relevant_Sentences'] = baseline.loc[mask_baseline_revised, 'PubMedID'].map(upd_map_rs)

    # Append NEW rows
    if not new_rows.empty:
        baseline = pd.concat([baseline, new_rows], ignore_index=True)

    # Drop duplicates by PubMedID keeping the last (so new/latest wins)
    before_drop = baseline.shape[0]
    baseline = baseline.drop_duplicates(subset='PubMedID', keep='last').reset_index(drop=True)
    after_drop = baseline.shape[0]

    #### UNIT CHECKS / ASSERTIONS ####
    # 1) no duplicates remain
    assert baseline['PubMedID'].nunique() == baseline.shape[0], "Duplicates remain after drop_duplicates!"

    # 2) baseline unique count should be >= previous unique (can't lose PubMedIDs)
    unique_after = baseline['PubMedID'].nunique()
    if unique_after < unique_before:
        raise AssertionError("Unexpected loss of PubMedID entries after update!")

    # 3) expected growth = new ids that were not previously present
    # some "new" ids could already have been present if updates file duplicated; check at least no unexpected shrink
    expected_min = unique_before
    if unique_after < expected_min:
        raise AssertionError("Baseline unique count decreased unexpectedly.")

    log.info("Baseline rows: before=%d after=%d (dropped %d duplicates)", baseline_before, baseline.shape[0], before_drop - after_drop)

    # Backup baseline before overwriting
    if backup and baseline_path.exists():
        backup_file(baseline_path)

    #### SAVING FINAL BASELINE DATASET MODIFIED ####
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(baseline_path, index=False)
    log.info("Updated baseline written to: %s", baseline_path)

    # Return final DataFrame
    return baseline


# ---------- CLI / Orchestration ----------
def parse_args():
    p = argparse.ArgumentParser(description="PubMed updates pipeline")
    p.add_argument("--no-download", action="store_true", help="Skip download step (assume files already present)")
    p.add_argument("--no-transform", action="store_true", help="Skip XML->CSV transform step")
    p.add_argument("--no-filter", action="store_true", help="Skip filter step (assume update CSV already present)")
    p.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of transform workers (processes)")
    p.add_argument("--threads", type=int, default=NUM_THREADS, help="Number of download threads")
    p.add_argument("--baseline", type=str, default=str(BASELINE_FILE_NAME), help="Baseline CSV path")
    p.add_argument("--updates", type=str, default=str(UPDATES_FILE_NAME), help="Updates CSV path produced by filter")
    return p.parse_args()


def main_cli():
    args = parse_args()

    # Apply CLI overrides to module-level settings
    transf.NUM_WORKERS = max(1, args.workers)
    extr.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transf.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filter.OUT_FOLDER = str(Path(args.updates).parent)

    baseline_path = Path(args.baseline)
    updates_path = Path(args.updates)

    # Load baseline
    try:
        baseline_df = safe_read_csv(baseline_path)
        log.info("Loaded baseline: %s", baseline_path)
    except FileNotFoundError:
        log.error("Baseline file not found: %s. Please run baseline extraction first.", baseline_path)
        return
    except Exception as e:
        log.exception("Failed loading baseline: %s", e)
        return

    # Run pipeline steps
    if not args.no_download:
        updates_extraction(num_threads=args.threads)
    else:
        log.info("Skipping download step (--no-download)")

    if not args.no_transform:
        xml_to_csv()
    else:
        log.info("Skipping transform step (--no-transform)")

    if not args.no_filter:
        run_filter()
    else:
        log.info("Skipping filter step (--no-filter)")

    # Load update CSV
    try:
        update_df = safe_read_csv(updates_path)
        log.info("Loaded updates: %s (rows=%d uniq_pubmed=%d)", updates_path, update_df.shape[0], update_df['PubMedID'].nunique())
    except FileNotFoundError:
        log.error("Updates file not found: %s. Aborting update step.", updates_path)
        return
    except Exception as e:
        log.exception("Failed loading updates file: %s", e)
        return

    # Run compare & update (with backup semantics)
    updated_baseline = compare_and_update(
        baseline_df,
        update_df,
        baseline_path=baseline_path,
        backup=True
    )

    log.info("Pipeline finished!")


if __name__ == "__main__":
    main_cli()
