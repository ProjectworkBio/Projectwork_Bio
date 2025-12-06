import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# === Load protein synonyms ===
syn_df = pd.read_csv("protein_synonyms.csv")

# Create mapping: canonical name -> list of synonyms (lowercase for fast search)
protein_synonyms = {
    row["Protein"]: [s.strip().lower() for s in str(row["Synonyms"]).split(";") if s.strip()]
    for _, row in syn_df.iterrows()
}

# Flatten all possible protein terms for search
all_terms = set()
for syns in protein_synonyms.values():
    all_terms.update(syns)

print(f"✅ Loaded {len(protein_synonyms)} proteins with {len(all_terms)} total synonyms.")


def process_file(file_path):
    matches = []
    filename = os.path.basename(file_path)

    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            try:
                tree = ET.parse(f)
            except ET.ParseError as e:
                print(f"⚠️ XML parse error in {filename}: {e}")
                return 0

            root = tree.getroot()
            for article in root.findall(".//PubmedArticle"):
                lang = article.findtext(".//Language")
                if lang != "eng":
                    continue

                # Extract abstract only (no title)
                abstracts = [abst.text for abst in article.findall(".//Abstract/AbstractText") if abst.text]
                if not abstracts:
                    continue
                abstract_text = " ".join(abstracts)
                abstract_lower = abstract_text.lower()

                # Find all matching protein terms
                matched_proteins = []
                for prot, syns in protein_synonyms.items():
                    if any(syn in abstract_lower for syn in syns):
                        matched_proteins.append(prot)

                if not matched_proteins:
                    continue

                pubmed_id = article.findtext(".//ArticleId[@IdType='pubmed']")
                matches.append({
                    "PubMedID": pubmed_id,
                    "Matched_Proteins": "; ".join(sorted(set(matched_proteins))),
                    "Abstract": abstract_text
                })

    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")
        return 0

    # Save matches
    if matches:
        os.makedirs("Result-v3", exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + "_matches.csv"
        output_path = os.path.join("Result-v3", output_filename)
        df = pd.DataFrame(matches)
        df.to_csv(output_path, index=False)
        print(f"✅ {filename}: {len(matches)} matches saved to {output_filename}")
        return len(matches)

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    data_folder = "Data"
    gz_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".gz")]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, gz_files))

    print("\n✅ All files processed!")
    print("Matches per file:", results)
