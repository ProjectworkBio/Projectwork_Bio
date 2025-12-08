import os
import pandas as pd
from collections import Counter

RESULT_FOLDER = "Result-v1"   # Folder with *_filtered.csv files
OUTPUT_FILE = "protein_frequency_stats.csv"

all_files = [
    os.path.join(RESULT_FOLDER, f)
    for f in os.listdir(RESULT_FOLDER)
    if f.endswith("_filtered.csv")
]

if not all_files:
    raise RuntimeError("No *_filtered.csv files found in Result folder!")

protein_counter = Counter()
total_abstracts = 0

for file in all_files:
    df = pd.read_csv(file)

    for proteins in df["Matched_Chemicals"].dropna():
        protein_list = [p.strip() for p in proteins.split(";") if p.strip()]

        for protein in protein_list:
            protein_counter[protein] += 1

    total_abstracts += len(df)

stats_df = pd.DataFrame(
    protein_counter.items(),
    columns=["Protein", "Occurrence_Count"]
).sort_values(by="Occurrence_Count", ascending=False)

stats_df.to_csv(OUTPUT_FILE, index=False)

print("\nPROTEIN OCCURRENCE STATISTICS COMPLETE")
print(f"Total abstracts analyzed: {total_abstracts}")
print(f"Total unique proteins: {len(stats_df)}")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nTOP 20 MOST FREQUENT PROTEINS:")
print(stats_df.head(20).to_string(index=False))
