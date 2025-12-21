import os
import polars as pl

result_folder = "Result-hybrid"
output_file = os.path.join(result_folder, "all_results_concatenated.csv")

# Collect all _sentences.csv files
csv_files = [
    os.path.join(result_folder, f)
    for f in os.listdir(result_folder)
    if f.endswith("_partial.csv")
]

if not csv_files:
    raise RuntimeError("No *_sentences.csv files found in Result folder.")

print(f"Found {len(csv_files)} CSV files")

# Read all CSV files
dfs = [pl.read_csv(f) for f in csv_files]

# Concatenate — Polars does this extremely reliably
combined = pl.concat(dfs, how="vertical")

# Ensure consistent column order (optional but recommended)
combined = combined.select([
    "PubMedID",
    "Matched_Protein_Id",
    "Matched_Protein_Name",
    "Other_protein",
    "Sentence",
])

# Save output
combined.write_csv(output_file)

print(f"Saved concatenated file: {output_file}")
print(f"Total rows: {combined.height}")
