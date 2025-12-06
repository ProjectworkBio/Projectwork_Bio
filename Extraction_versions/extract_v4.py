import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sentence_splitter import SentenceSplitter, split_text_into_sentences

# Load protein synonyms
syn_df = pd.read_csv("protein_synonyms.csv")

# Normalize a synonym (lowercase, hyphen/space-insensitive)
def normalize_synonym(s):
    return re.sub(r'[-\s]+', '', s.strip().lower())

# Create mapping: canonical name -> list of normalized synonyms
protein_synonyms = {}
for _, row in syn_df.iterrows():
    all_names = [row["Protein"]] + str(row["Synonyms"]).replace("\t", " ").split(";")
    cleaned = list(set(normalize_synonym(s) for s in all_names if s.strip()))
    protein_synonyms[row["Protein"].strip()] = cleaned

# Flatten for info
all_terms = {s for syns in protein_synonyms.values() for s in syns}
print(f"✅ Loaded {len(protein_synonyms)} proteins with {len(all_terms)} total normalized synonyms.")


# === Sentence splitting regex ===
# sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

splitter = SentenceSplitter(language='en')

# Normalize text for flexible matching (case-insensitive, remove hyphens/spaces)
def normalize_text(text):
    return re.sub(r'[-\s]+', '', text.lower())

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

                # Extract abstract text
                abstracts = [abst.text for abst in article.findall(".//Abstract/AbstractText") if abst.text]
                if not abstracts:
                    continue
                abstract_text = " ".join(abstracts)
                # sentences = re.split(sentence_splitter, abstract_text)
                sentences = splitter.split(abstract_text)

                relevant_sentences = []
                proteins_in_abstract = set()

                for sent in sentences:
                    sent_norm = normalize_text(sent)
                    matched = set()

                    for prot, syns in protein_synonyms.items():
                        if any(re.search(rf'\b{re.escape(syn)}\b', sent_norm) for syn in syns):
                            matched.add(prot)
                    if len(matched) >= 2:
                        relevant_sentences.append(sent.strip())
                        proteins_in_abstract.update(matched)

                if not relevant_sentences:
                    continue  # skip abstracts without ≥2-protein sentences

                pubmed_id = article.findtext(".//ArticleId[@IdType='pubmed']")
                matches.append({
                    "PubMedID": pubmed_id,
                    "Matched_Proteins": "; ".join(sorted(proteins_in_abstract)),
                    "Abstract": abstract_text.strip(),
                    "Relevant_Sentences": " || ".join(relevant_sentences)
                })

    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")
        return 0

    # Save matches
    if matches:
        os.makedirs("Result-v4", exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + "_2prot_sentences.csv"
        output_path = os.path.join("Result-v4", output_filename)
        df = pd.DataFrame(matches)
        df.to_csv(output_path, index=False)
        print(f"✅ {filename}: {len(matches)} abstracts saved to {output_filename}")
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
