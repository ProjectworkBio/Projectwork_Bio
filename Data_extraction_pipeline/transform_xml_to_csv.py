import os
import gzip
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

INPUT_DIR = Path("Pubmed_Baseline")
OUTPUT_DIR = Path("Pubmed_abstracts_csvs")
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = os.cpu_count()

# Worker function: Extract PubMedID and AbstractText from a .xml.gz, write to CSV. Returns the filename if successful.
def extract_pubmed_data(xml_gz_path, csv_output_path):
    try:
        with gzip.open(xml_gz_path, "rb") as f:
            context = ET.iterparse(f, events=("end",))

            with open(csv_output_path, "w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(["PubMedID", "AbstractText"])

                for event, elem in context:
                    if elem.tag == "PubmedArticle":

                        # Get PMID
                        pmid_el = elem.find(".//PMID")
                        pmid = pmid_el.text.strip() if pmid_el is not None else None

                        # Get AbstractText (some articles have multiple sections)
                        abstract_text_elements = elem.findall(".//AbstractText")

                        if not abstract_text_elements:
                            elem.clear()
                            continue

                        full_abstract = " ".join(
                            (t.text or "").replace("\n", " ").strip()
                            for t in abstract_text_elements
                        ).strip()

                        if pmid and full_abstract:
                            writer.writerow([pmid, full_abstract])

                        elem.clear()

            del context
            return xml_gz_path.name, "ok"

    except Exception as e:
        return xml_gz_path.name, f"error: {e}"


def main():
    files = sorted(INPUT_DIR.glob("*.xml.gz"))

    if not files:
        print("No .xml.gz files found in Pubmed_Baseline")
        return

    print(f"Found {len(files)} files. Processing with {NUM_WORKERS} workers...\n")

    futures = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        for xml_file in files:
            out_file = OUTPUT_DIR / (xml_file.stem.replace(".xml", "") + ".csv")
            # Skip already processed files
            if out_file.exists():
                continue

            futures.append(
                executor.submit(extract_pubmed_data, xml_file, out_file)
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing XML files"):
            pass

    # Summary
    ok = 0
    failed = []
    for fut in futures:
        fname, status = fut.result()
        if status == "ok":
            ok += 1
        else:
            failed.append((fname, status))

    print(f"\nFinished. Successfully processed: {ok}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nErrors:")
        for f, err in failed:
            print(f"  {f}: {err}")


if __name__ == "__main__":
    main()
