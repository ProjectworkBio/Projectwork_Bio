import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sentence_splitter import SentenceSplitter

def load_synonyms(csv_path):
    df = pd.read_csv(csv_path)
    synonym_data = {}

    for _, row in df.iterrows():
        name = str(row["ProteinName"]).strip()
        uniprot_id = str(row.get("UniProtID", "")).strip()
        synonyms = set()
        synonyms.add(name)

        # Protein synonyms
        if pd.notna(row.get("ProteinSynonyms")):
            for s in str(row["ProteinSynonyms"]).split(","):
                synonyms.add(s.strip())

        # Gene name
        if pd.notna(row.get("GeneName")):
            synonyms.add(str(row["GeneName"]).strip())

        # Gene synonyms
        if pd.notna(row.get("GeneSynonyms")):
            for s in str(row["GeneSynonyms"]).split(","):
                synonyms.add(s.strip())

        patterns = []
        for syn in synonyms:
            syn_escaped = re.escape(syn)
            # \b matches word boundaries, but also handle parentheses/brackets
            pattern = rf'(?<!\w){syn_escaped}(?!\w)'
            patterns.append(pattern)

        synonym_data[name] = {
            "uniprot": uniprot_id,
            "synonyms": synonyms,
            "regex": re.compile("|".join(patterns), re.IGNORECASE)
        }

    return synonym_data

synonym_data = load_synonyms("protein_synonyms.csv")
splitter = SentenceSplitter(language="en")

def process_csv(file_path):
    filename = os.path.basename(file_path)
    print(f"Processing {filename} ...")

    df = pd.read_csv(file_path)
    results = []

    for _, row in df.iterrows():
        pmid = row["PubMedID"]
        abstract = str(row["AbstractText"])
        sentences = splitter.split(abstract)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            matched_synonyms = []
            matched_uniprot_ids = []

            for _, info in synonym_data.items():
                matches = info["regex"].findall(sentence_clean)
                if matches:
                    matched_synonyms.extend(matches)
                    matched_uniprot_ids.append(info["uniprot"])

            if matched_synonyms:
                results.append({
                    "PubMedID": pmid,
                    "Matched_Proteins_UniProtId": ";".join(matched_uniprot_ids),
                    "Matched_Synonym": ";".join(matched_synonyms),
                    "Sentence": sentence_clean
                })

    if results:
        os.makedirs("Result-v3", exist_ok=True)
        outname = os.path.splitext(filename)[0] + "_sentences.csv"
        outpath = os.path.join("Result-v3", outname)
        pd.DataFrame(results).to_csv(outpath, index=False)
        print(f"Saved {len(results)} sentences → {outname}")
        return len(results)

    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()

    data_folder = "PubMed_abstracts_csvs"
    input_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".csv")
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_csv, input_files))

    print("\nDone!")
    print("Matches per file:", results)
