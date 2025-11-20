import os
import io
import datetime
import requests
import gzip
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from extract_v5 import process_pubmed_file

URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
OUT_FOLDER = "Result-updates"
N_WORKERS = 4


def process_files(xml_contents):
    """Receives a list of (filename, xml_bytes)"""
    all_rows = []

    def worker(xml_bytes):
        try:
            return process_pubmed_file(xml_bytes)
        except Exception as e:
            print(f"[ERROR worker] {e}")
            return []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = [exe.submit(worker, xml_bytes) for (_, xml_bytes) in xml_contents]

        for f in futures:
            try:
                rows = f.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                print(f"[ERROR] {e}")

    os.makedirs(OUT_FOLDER, exist_ok=True)
    filename = os.path.join(
        OUT_FOLDER,
        f"updates_{datetime.date.today()}.csv"
    )

    pd.DataFrame(all_rows).to_csv(filename, index=False)
    print(f"\nDONE. Extracted {len(all_rows)} rows → {filename}")

    return filename


def obtain_files():
    """Returns a list of tuples:(filename, xml_bytes)"""
    resp = requests.get(URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".xml.gz")]
    print(f"Found {len(links)} files.")

    collected = []

    for filename in links:
        file_url = f"{URL}{filename}"
        print("Downloading:", file_url)

        r = requests.get(file_url, stream=True)
        r.raise_for_status()

        compressed = io.BytesIO(r.content)

        with gzip.GzipFile(fileobj=compressed) as gz:
            xml_bytes = gz.read()

        collected.append((filename, xml_bytes))

    return collected


def compare_files(original, updates):

    # Merge on PubMedID
    merged = updates.merge(
        original[['PubMedID', 'Matched_Proteins', 'Relevant_Sentences']],
        on='PubMedID',
        how='left',
        suffixes=('_new', '_old')
    )

    # 1) NEW rows:
    new_rows = merged[merged['Matched_Proteins_old'].isna()][[
        'PubMedID', 'Matched_Proteins_new', 'Relevant_Sentences_new'
    ]].rename(columns={
        'Matched_Proteins_new': 'Matched_Proteins',
        'Relevant_Sentences_new': 'Relevant_Sentences'
    })

    # 2) REVISED rows:
    updated_rows = merged[
        (merged['Matched_Proteins_old'].notna()) & 
        (
            (merged['Matched_Proteins_new'] != merged['Matched_Proteins_old']) |
            (merged['Relevant_Sentences_new'] != merged['Relevant_Sentences_old'])
        )
    ]

    # Apply updates to original
    for _, row in updated_rows.iterrows():
        original.loc[original['PubMedID'] == row['PubMedID'], 'Matched_Proteins'] = row['Matched_Proteins_new']
        original.loc[original['PubMedID'] == row['PubMedID'], 'Relevant_Sentences'] = row['Relevant_Sentences_new']

    # Append NEW rows
    original = pd.concat([original, new_rows], ignore_index=True)

    # Upload the modified dataset:
    #original.to_csv('updates.csv', index=False)

    print("Updates processed!")
    return original


def main():
    print("Downloading XML.gz...")
    xml_contents = obtain_files()

    print("Processing files...")
    updates_file = process_files(xml_contents)

    print("Comparing with baseline...")
    baseline = pd.read_csv("relevant_sentences.csv")
    updates = pd.read_csv(updates_file)

    output = compare_files(baseline, updates)
    output.to_csv("relevant_sentences.csv", index=False)


if __name__ == '__main__':
    main()