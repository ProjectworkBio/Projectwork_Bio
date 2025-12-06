import os
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Protein to MeSH mapping
protein_mesh_mapping = {
    "Interleukin-18": ("IL18", "D020382"),
    "Hepatocyte growth factor": ("HGF", "D017228"),
    "C-C motif chemokine 19": ("CCL19", "D018451"),
    "C-C motif chemokine 2": ("CCL2", "D018447"),
    "Macrophage metalloelastase": ("MMP12", "D053508"),
    "Lymphotoxin-alpha": ("LTA", "D008233"),
    "Fms-related tyrosine kinase 3 ligand": ("FLT3LG", "D051941"),
    "Tumor necrosis factor": ("TNF", "D014409"),
    "Interleukin-17A": ("IL17A", "D020381"),
    "Interleukin-17F": ("IL17F", "D020381"),
    "Interleukin-17C": ("IL17C", "D020381"),
    "Interleukin-2": ("IL2", "D007376"),
    "Granulocyte colony-stimulating factor": ("CSF3", "D016179"),
    "Interleukin-1 beta": ("IL1B", "D053583"),
    "Oxidized low-density lipoprotein receptor 1": ("OLR1", "D051127"),
    "Tumor necrosis factor ligand superfamily member 12": ("TNFSF12", "D000074049"),
    "C-X-C motif chemokine 10": ("CXCL10", "D018481"),
    "Vascular endothelial growth factor A": ("VEGFA", "D042461"),
    "Interleukin-33": ("IL33", "D000067596"),
    "Thymic stromal lymphopoietin": ("TSLP", "D000094632"),
    "Interferon gamma": ("IFNG", "D015458"),
    "C-C motif chemokine 4": ("CCL4", "D018449"),
    "Protransforming growth factor alpha": ("TGFA", "D013923"),
    "Interleukin-13": ("IL13", "D018793"),
    "Interleukin-8": ("CXCL8", "D016209"),
    "C-C motif chemokine 8": ("CCL8", "D018450"),
    "Interleukin-6": ("IL6", "D015850"),
    "C-C motif chemokine 13": ("CCL13", "D019214"),
    "Granulocyte-macrophage colony-stimulating factor": ("CSF2", "D016178"),
    "C-C motif chemokine 7": ("CCL7", "D018451"),
    "Interleukin-4": ("IL4", "D015847"),
    "Tumor necrosis factor ligand superfamily member 10": ("TNFSF10", "D053221"),
    "Oncostatin-M": ("OSM", "D009816"),
    "Interstitial collagenase": ("MMP1", "D020781"),
    "Pro-epidermal growth factor": ("EGF", "D010048"),
    "Interleukin-7": ("IL7", "D015851"),
    "Interleukin-15": ("IL15", "D019409"),
    "Macrophage colony-stimulating factor 1": ("CSF1", "D015441"),
    "C-X-C motif chemokine 9": ("CXCL9", "D018480"),
    "C-X-C motif chemokine 11": ("CXCL11", "D018482"),
    "Stromal cell-derived factor 1": ("CXCL12", "D054377"),
    "Eotaxin": ("CCL11", "D004740"),
    "Interleukin-10": ("IL10", "D016753"),
    "C-C motif chemokine 3": ("CCL3", "D018448"),
    "Interleukin-27": ("EBI3_IL27", "D064094")
}

# Mapping UI -> protein names
ui_to_proteins = {}
for protein, (_, ui) in protein_mesh_mapping.items():
    ui_to_proteins.setdefault(ui, []).append(protein)


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

                chemicals = []
                for chem in article.findall(".//Chemical"):
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
                    continue

                abstract_texts = [
                    abst.text.strip()
                    for abst in article.findall(".//Abstract/AbstractText")
                    if abst.text and abst.text.strip()
                ]
                if not abstract_texts:
                    continue

                abstract = " ".join(abstract_texts)
                pubmed_id = article.findtext(".//ArticleId[@IdType='pubmed']")

                matches.append({
                    "PubMedID": pubmed_id,
                    "Matched_Chemicals": "; ".join(matched_proteins),
                    "Matched_UI": "; ".join(matched_uis),
                    "Abstract": abstract
                })

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0

    if matches:
        os.makedirs("Result", exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + "_filtered.csv"
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
