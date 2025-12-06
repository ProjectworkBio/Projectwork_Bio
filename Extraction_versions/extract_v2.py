import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Protein list
proteins = [
    "Interleukin-18", "Hepatocyte growth factor", "C-C motif chemokine 19",
    "C-C motif chemokine 2", "Macrophage metalloelastase", "Lymphotoxin-alpha",
    "Fms-related tyrosine kinase 3 ligand", "Tumor necrosis factor",
    "Interleukin-17A", "Interleukin-17F", "Interleukin-17C", "Interleukin-2",
    "Granulocyte colony-stimulating factor", "Interleukin-1 beta",
    "Oxidized low-density lipoprotein receptor 1",
    "Tumor necrosis factor ligand superfamily member 12",
    "C-X-C motif chemokine 10", "Vascular endothelial growth factor A",
    "Interleukin-33", "Thymic stromal lymphopoietin", "Interferon gamma",
    "C-C motif chemokine 4", "Protransforming growth factor alpha",
    "Interleukin-13", "Interleukin-8", "C-C motif chemokine 8", "Interleukin-6",
    "C-C motif chemokine 13", "Granulocyte-macrophage colony-stimulating factor",
    "C-C motif chemokine 7", "Interleukin-4",
    "Tumor necrosis factor ligand superfamily member 10", "Oncostatin-M",
    "Interstitial collagenase", "Pro-epidermal growth factor",
    "Interleukin-7", "Interleukin-15", "Macrophage colony-stimulating factor 1",
    "C-X-C motif chemokine 9", "C-X-C motif chemokine 11", "Interleukin-17C",
    "Stromal cell-derived factor 1", "Eotaxin", "Interleukin-10",
    "C-C motif chemokine 3", "Interleukin-27"
]

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

                # Combine title and abstract for searching
                texts = []
                title = article.findtext(".//ArticleTitle")
                if title:
                    texts.append(title)
                abstracts = [abst.text for abst in article.findall(".//Abstract/AbstractText") if abst.text]
                texts.extend(abstracts)
                combined_text = " ".join(texts).lower()

                matched_proteins = [p for p in proteins if p.lower() in combined_text]
                if not matched_proteins:
                    continue

                abstract_text = " ".join(abstracts) if abstracts else ""
                pubmed_id = article.findtext(".//ArticleId[@IdType='pubmed']")

                matches.append({
                    "PubMedID": pubmed_id,
                    "Matched_Proteins": "; ".join(matched_proteins),
                    "Abstract": abstract_text
                })

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0

    if matches:
        os.makedirs("Result", exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + "_proteins.csv"
        output_path = os.path.join("Result", output_filename)
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
