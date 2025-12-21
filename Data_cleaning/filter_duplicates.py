import pandas as pd

# Load the protein synonyms dictionary
df_dict = pd.read_csv("protein_synonyms.csv")

# Build a dictionary mapping ProteinName -> set of all synonyms
protein_syn_dict = {}
for _, row in df_dict.iterrows():
    protein_name = str(row["ProteinName"]).strip()
    synonyms = set()

    # Always include the primary protein name
    if protein_name:
        synonyms.add(protein_name)

    # Include ProteinSynonyms
    if pd.notna(row.get("ProteinSynonyms")):
        synonyms.update(
            s.strip() for s in row["ProteinSynonyms"].split(",") if s.strip()
        )

    # Include GeneName
    if pd.notna(row.get("GeneName")):
        gene_name = str(row["GeneName"]).strip()
        if gene_name:
            synonyms.add(gene_name)

    # Include GeneSynonyms
    if pd.notna(row.get("GeneSynonyms")):
        synonyms.update(
            s.strip() for s in row["GeneSynonyms"].split(",") if s.strip()
        )

    protein_syn_dict[protein_name] = synonyms

# Load the results file
df_results = pd.read_csv("Result-hybrid\\all_results_concatenated.csv")

# Function to check if Other_protein is a synonym of Matched_Protein_Name
def is_synonym(row):
    matched_name = row["Matched_Protein_Name"]
    other = row["Other_protein"]
    synonyms = protein_syn_dict.get(matched_name, set())
    return other in synonyms

# Boolean mask for rows to delete
mask_delete = df_results.apply(is_synonym, axis=1)

# Count how many rows will be deleted
num_deleted = mask_delete.sum()
print(f"Number of rows deleted: {num_deleted}")

# Filter out the rows
df_filtered = df_results[~mask_delete]

# Save the filtered results
df_filtered.to_csv("Result-hybrid\\results_filtered.csv", index=False)
