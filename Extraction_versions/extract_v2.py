import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sentence_splitter import SentenceSplitter

def load_proteins(csv_path):
    df = pd.read_csv(csv_path)
    protein_dict = {}
    for _, row in df.iterrows():
        uni = str(row["UniProtID"]).strip()
        name = str(row["ProteinName"]).strip()

        protein_dict[name.lower()] = {
            "uniprot": uni,
            "name": name
        }
    return protein_dict

protein_data = load_proteins("protein_synonyms.csv")
splitter = SentenceSplitter(language="en")

def process_csv(file_path):
    filename = os.path.basename(file_path)
    print(f"Processing {filename} ...")

    df = pd.read_csv(file_path)
    matches = []

    # Precompute lowercase protein names for fast matching
    protein_names_lower = list(protein_data.keys())

    for _, row in df.iterrows():
        pmid = row["PubMedID"]
        abstract = str(row["AbstractText"])

        # Split into sentences
        sentences = splitter.split(abstract)

        for sentence in sentences:
            sent_lower = sentence.lower()

            # Store found proteins for this sentence
            found_uniprot_ids = []
            found_names = []

            for prot_lower in protein_names_lower:
                if prot_lower in sent_lower:
                    info = protein_data[prot_lower]
                    found_uniprot_ids.append(info["uniprot"])
                    found_names.append(info["name"])

            # If any proteins matched → store ONE row per sentence
            if found_uniprot_ids:
                # if "Q99616" in found_uniprot_ids:
                    matches.append({
                        "PubMedId": pmid,
                        "Matched_Proteins_UniProtId": ";".join(found_uniprot_ids),
                        "Matched_Proteins_Name": ";".join(found_names),
                        "Sentence": sentence.strip()
                    })

    if matches:
        os.makedirs("Result-v2", exist_ok=True)
        outname = os.path.splitext(filename)[0] + "_sentences.csv"
        outpath = os.path.join("Result-v2", outname)
        pd.DataFrame(matches).to_csv(outpath, index=False)
        print(f"Saved {len(matches)} sentences to {outname}")
        return len(matches)

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
