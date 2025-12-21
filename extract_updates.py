import os
import io
import datetime
import requests
import gzip
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
OUT_FOLDER = "Result-updates"


def extract_pubmed_articles_from_gz(stream):
    """
    Stream-parse a PubMed XML.gz file using iterparse.
    Yields rows one-by-one to avoid memory explosion.
    """
    with gzip.GzipFile(fileobj=stream) as gz:
        context = ET.iterparse(gz, events=("end",))
        
        for event, elem in context:
            if elem.tag == "PubmedArticle":

                pmid = elem.findtext(".//ArticleId[@IdType='pubmed']")
                if pmid is None:
                    elem.clear()
                    continue

                abstracts = [
                    ab.text for ab in elem.findall(".//Abstract/AbstractText")
                    if ab.text
                ]
                if abstracts:
                    abstract_text = " ".join(abstracts)
                    yield {
                        "PubMedID": pmid,
                        "Abstract": abstract_text
                    }

                # Free memory for processed XML subtree
                elem.clear()


def obtain_and_process_files():
    """Download PubMed updatefiles and process them safely."""
    resp = requests.get(URL)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)
             if a["href"].endswith(".xml.gz")]

    print(f"Found {len(links)} files.")

    os.makedirs(OUT_FOLDER, exist_ok=True)

    out_file = os.path.join(
        OUT_FOLDER,
        f"updates_{datetime.date.today()}.csv"
    )

    # Open output CSV once — append as we go
    with open(out_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PubMedID", "Abstract"])
        writer.writeheader()

        total_rows = 0

        for filename in links:
            file_url = f"{URL}{filename}"
            print(f"\nDownloading {file_url}")

            r = requests.get(file_url, stream=True)
            r.raise_for_status()

            compressed_stream = io.BytesIO(r.content)

            # Stream parse and write rows incrementally
            for row in extract_pubmed_articles_from_gz(compressed_stream):
                writer.writerow(row)
                total_rows += 1

            print(f" → Finished {filename}, total rows so far: {total_rows}")

    print(f"\nDONE. Extracted {total_rows} rows → {out_file}")
    return out_file


def final_process():
    ...


def main():
    print("Streaming & processing large XML.gz files safely...")
    obtain_and_process_files()


if __name__ == "__main__":
    main()
