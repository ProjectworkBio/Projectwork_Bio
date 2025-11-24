import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
OUTPUT_DIR = Path("Pubmed_Baseline")
NUM_THREADS = 10   # Increase for faster download
# Regex pattern matching ALL baseline PubMed files:
#   pubmedXXnYYYY.xml.gz → XX = year, YYYY = file number
FILENAME_PATTERN = re.compile(r"pubmed\d{2}n\d{4}\.xml\.gz")


#Fetch directory listing from NCBI FTP and extract all pubmed XML.gz filenames.
def list_files(): 
    r = requests.get(BASE_URL)
    r.raise_for_status()
    files = FILENAME_PATTERN.findall(r.text)
    return sorted(set(files))


def download_file(filename):
    url = BASE_URL + filename
    out_path = OUTPUT_DIR / filename

    # Skip existing file
    if out_path.exists():
        return filename, "exists"

    # Streaming download
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))

            with open(out_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
                leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return filename, "downloaded"

    except Exception as e:
        return filename, f"error: {e}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Fetching file list from PubMed baseline FTP...")
    files = list_files()
    print(f"Found {len(files)} files to download.\n")

    # Parallel download
    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(download_file, f): f for f in files}

        for fut in tqdm(as_completed(futures), total=len(files), desc="Overall progress"):
            results.append(fut.result())

    # Summary
    downloaded = sum(1 for _, status in results if status == "downloaded")
    skipped = sum(1 for _, status in results if status == "exists")
    failed = [f for f, status in results if status.startswith("error")]

    print("\n===== SUMMARY =====")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(" -", f)


if __name__ == "__main__":
    main()
