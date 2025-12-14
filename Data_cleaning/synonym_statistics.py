import pandas as pd
from collections import defaultdict

# Load CSVs
results_df = pd.read_csv("Result-v5\\results_protein_matches.csv")
synonyms_df = pd.read_csv("protein_synonyms.csv")

# Create a dictionary: synonym -> set of UniProtIDs
synonym_to_uniprot = defaultdict(set)
for _, row in synonyms_df.iterrows():
    synonym = row['ProteinName'].strip().lower()
    uniprot = row['UniProtID'].strip()
    synonym_to_uniprot[synonym].add(uniprot)

# Initialize counts
false_positive_counts = defaultdict(int)
total_counts = defaultdict(int)

# Check each matched protein in results
for _, row in results_df.iterrows():
    matched_proteins = [p.strip().lower() for p in str(row['Matched_Proteins']).split(';')]
    predicted_uniprots = [u.strip() for u in str(row['UniProtIDs']).split(';')]
    
    for synonym, predicted in zip(matched_proteins, predicted_uniprots):
        total_counts[synonym] += 1
        # If the predicted UniProtID is not in the dictionary for this synonym → false positive
        if predicted not in synonym_to_uniprot.get(synonym, set()):
            false_positive_counts[synonym] += 1

# Calculate false positive rate
fp_rate = {syn: false_positive_counts[syn] / total_counts[syn] for syn in total_counts}

# Create a ranked dataframe
fp_df = pd.DataFrame({
    'Synonym': list(fp_rate.keys()),
    'False_Positive_Rate': list(fp_rate.values()),
    'Total_Matches': [total_counts[syn] for syn in total_counts],
    'False_Positives': [false_positive_counts[syn] for syn in total_counts]
})

fp_df = fp_df.sort_values(by='False_Positive_Rate', ascending=False)

# Save results
fp_df.to_csv("synonym_false_positive_ranking.csv", index=False)

print(fp_df.head(20))  # Top 20 synonyms with highest false positive rate
