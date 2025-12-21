# This extraction version utilizes the <Chemical> XML elements of the PubMed dataset. 

import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

# Load protein to MeSH UI mapping
def load_protein_mesh(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    ui_to_proteins = {}
    for _, row in df.iterrows():
        ui = str(row["UI"]).strip()
        protein = str(row["Protein name"]).strip()
        if ui not in ui_to_proteins:
            ui_to_proteins[ui] = []
        ui_to_proteins[ui].append(protein)
    return ui_to_proteins

ui_to_proteins = load_protein_mesh(Path("Helper_functions") / "protein_mesh.csv")

def process_file(file_path):
    matches = []
    filename = os.path.basename(file_path)

    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            context = ET.iterparse(f, events=("end",))

            for event, elem in context:

                # Process ONLY article nodes
                if elem.tag.endswith("PubmedArticle"):

                    lang = elem.findtext(".//Language")
                    if lang != "eng":
                        elem.clear()
                        continue

                    # Extract Chemicals
                    chemicals = []
                    for chem in elem.findall(".//Chemical"):
                        name_el = chem.find("NameOfSubstance")
                        ui = name_el.attrib.get("UI") if name_el is not None else None
                        text = name_el.text.strip() if name_el is not None and name_el.text else None
                        if ui:
                            chemicals.append((text, ui))

                    matched_proteins = []
                    matched_uis = []

                    for text, ui in chemicals:
                        if ui in ui_to_proteins:
                            if ui == "D020381":  # special case for IL17 family
                                if text in ["Interleukin-17A", "Interleukin-17F", "Interleukin-17C"]:
                                    matched_proteins.append(text)
                                    matched_uis.append(ui)
                            else:
                                matched_proteins.extend(ui_to_proteins[ui])
                                matched_uis.append(ui)

                    if not matched_proteins:
                        elem.clear()
                        continue

                    abstract_texts = [
                        abst.text.strip()
                        for abst in elem.findall(".//Abstract/AbstractText")
                        if abst.text and abst.text.strip()
                    ]
                    if not abstract_texts:
                        continue

                    abstract = " ".join(abstract_texts)
                    pubmed_id = elem.findtext(".//ArticleId[@IdType='pubmed']")

                    matches.append({
                        "PubMedID": pubmed_id,
                        "Matched_Chemicals": "; ".join(matched_proteins),
                        "Matched_UI": "; ".join(matched_uis),
                        "Abstract": abstract
                    })
                    elem.clear()

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0

    if matches:
        os.makedirs("Result-v1", exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + "_filtered.csv"
        output_path = os.path.join("Result-v1", output_filename)
        df = pd.DataFrame(matches)
        df.to_csv(output_path, index=False)
        print(f"{filename}: {len(matches)} matches saved to {output_filename}")
        return len(matches)

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()

    data_folder = "Pubmed_Baseline"
    gz_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".gz") and f > "pubmed25n1120.xml.gz"]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, gz_files))

    print("\nAll files are processed.")     
    print("Matches per file:", results)
