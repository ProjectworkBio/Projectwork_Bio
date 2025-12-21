import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process 

# Load CSV
df = pd.read_csv(Path("Results") / "Results_ner_only" / "results_protein_matches.csv")

def check_match(sentence, protein, threshold=80):
    """
    Returns True if fuzzy similarity is above threshold.
    """
    score = fuzz.partial_ratio(protein.lower(), sentence.lower())
    return score, score >= threshold

results = []

for idx, row in df.iterrows():
    score, match = check_match(row["Sentence"], row["Matched_Proteins"])
    
    results.append({
        "PubMedID": row["PubMedID"],
        "Sentence": row["Sentence"],
        "Matched_Proteins": row["Matched_Proteins"],
        "UniProtIDs": row["UniProtIDs"],
        "Match_Score": score,
        "Match": match
    })

results_df = pd.DataFrame(results)

# Save results to CSV
output_file = Path ("Data_cleaning") / "protein_matching_results.csv"
results_df.to_csv(output_file, index=False)

print(f"\nSaved analysis to {output_file}\n")
print(results_df.head())
print("\nSummary:")
print("Matched counts:\n", results_df["Match"].value_counts())
print("Average Score:", results_df["Match_Score"].mean())
