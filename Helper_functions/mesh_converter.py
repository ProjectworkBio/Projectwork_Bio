
import xml.etree.ElementTree as ET
import csv

mesh_xml_file = "desc2025.xml"
output_csv = "protein_mesh.csv"

protein_list = [
    "Interleukin-18", "Hepatocyte growth factor", "C-C motif chemokine 19",
    "C-C motif chemokine 2", "Macrophage metalloelastase", "Lymphotoxin-alpha",
    "Fms-related tyrosine kinase 3 ligand", "Tumor necrosis factor", "Interleukin-17A",
    "Interleukin-2", "Interleukin-17F", "Granulocyte colony-stimulating factor",
    "Interleukin-1 beta", "Oxidized low-density lipoprotein receptor 1",
    "Tumor necrosis factor ligand superfamily member 12", "C-X-C motif chemokine 10",
    "Vascular endothelial growth factor A", "Interleukin-33", "Thymic stromal lymphopoietin",
    "Interferon gamma", "C-C motif chemokine 4", "Protransforming growth factor alpha",
    "Interleukin-13", "Interleukin-8", "C-C motif chemokine 8", "Interleukin-6",
    "C-C motif chemokine 13", "Granulocyte-macrophage colony-stimulating factor",
    "C-C motif chemokine 7", "Interleukin-4", "Tumor necrosis factor ligand superfamily member 10",
    "Oncostatin-M", "Interstitial collagenase", "Pro-epidermal growth factor", "Interleukin-7",
    "Interleukin-15", "Macrophage colony-stimulating factor 1", "C-X-C motif chemokine 9",
    "C-X-C motif chemokine 11", "Interleukin-17C", "Stromal cell-derived factor 1", "Eotaxin",
    "Interleukin-10", "C-C motif chemokine 3", "Interleukin-27"
]

protein_set = set(name.upper() for name in protein_list)
tree = ET.parse(mesh_xml_file)
root = tree.getroot()

mesh_term_to_ui = {}
for descriptor in root.findall(".//DescriptorRecord"):
    ui = descriptor.findtext("DescriptorUI")
    heading = descriptor.findtext("DescriptorName/String")
    if heading:
        mesh_term_to_ui[heading.upper()] = ui
    for concept in descriptor.findall(".//ConceptList/Concept"):
        for term in concept.findall(".//TermList/Term/String"):
            if term.text:
                mesh_term_to_ui[term.text.upper()] = ui

# Map proteins to MeSH UI

lookup_table = []
not_found_proteins = set(protein_list)

for protein in protein_list:
    ui = mesh_term_to_ui.get(protein.upper())
    if ui:
        lookup_table.append({
            "ProteinName": protein,
            "MeSHName": protein,
            "MeSH_UI": ui
        })
        not_found_proteins.discard(protein)

print(f"Saving results to {output_csv} ...")
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["ProteinName", "MeSHName", "MeSH_UI"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in lookup_table:
        writer.writerow(entry)