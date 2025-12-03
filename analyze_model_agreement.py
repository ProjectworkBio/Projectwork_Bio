import pandas as pd
import os
import numpy as np

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'analysis')
MODEL_DIR = os.path.join(BASE_DIR, 'results', 'reports')

MODEL_RESULTS = {
    "BioBERT":     os.path.join(MODEL_DIR, "ppi_report_v2_BioBERT.csv"),
    "PubMedBERT":  os.path.join(MODEL_DIR, "ppi_report_v2_PubMedBERT.csv"),
    "BioLinkBERT": os.path.join(MODEL_DIR, "ppi_report_v2_BioLinkBERT.csv"),
    "SciBert":   os.path.join(MODEL_DIR, "ppi_report_v2_SciBert.csv")
}

OUTPUT_COMPARISON = os.path.join(RESULTS_DIR, "full_model_comparison_v2.csv")
OUTPUT_DISAGREEMENTS = os.path.join(RESULTS_DIR, "model_disagreements_for_judge_v2.csv")

JOIN_KEYS = ['PubMedID', 'Entity_1_Raw', 'Entity_2_Raw', 'Sentence', 'Entity_1_Name', 'Entity_2_Name']

# DATA LOADING and MERGING
merged_df = None

print("--- Loading and Merging Results ---")

for model_name, file_path in MODEL_RESULTS.items():
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {model_name}: {file_path}")
        continue
    
    print(f"Loading {model_name}...")
    df = pd.read_csv(file_path)
    
    cols_to_keep = JOIN_KEYS + ['Original_Label', 'Confidence']
    df = df[cols_to_keep].copy()
    
    df.rename(columns={
        'Original_Label': f'{model_name}_Relation',
        'Confidence': f'{model_name}_Conf'
    }, inplace=True)
    
    # Merge
    if merged_df is None:
        merged_df = df
    else:
        # Inner join only looks at rows all models found
        merged_df = pd.merge(merged_df, df, on=JOIN_KEYS, how='inner')

print(f"\nAligned {len(merged_df)} common entity pairs across all models.")

# ANALYSIS LOGIC

rel_cols = [f'{m}_Relation' for m in MODEL_RESULTS.keys() if f'{m}_Relation' in merged_df.columns]
conf_cols = [f'{m}_Conf' for m in MODEL_RESULTS.keys() if f'{m}_Conf' in merged_df.columns]

def check_consensus(row):
    """Returns True if all models predict the exact same relation label."""
    preds = [row[c] for c in rel_cols]
    return len(set(preds)) == 1

def get_majority_vote(row):
    """Returns the label with the most votes."""
    preds = [row[c] for c in rel_cols]
    return max(set(preds), key=preds.count)

# Apply Logic
print("Analyzing Agreement...")
merged_df['All_Agree'] = merged_df.apply(check_consensus, axis=1)
merged_df['Majority_Vote'] = merged_df.apply(get_majority_vote, axis=1)

# Calculate Mean Confidence
merged_df['Avg_Confidence'] = merged_df[conf_cols].mean(axis=1)

# STATISTICS 

agreement_rate = (merged_df['All_Agree'].sum() / len(merged_df)) * 100
print(f"\nStatistics:")
print(f"Total Pairs Evaluated: {len(merged_df)}")
print(f"Full Consensus Rate:   {agreement_rate:.2f}%")

# Save Full Report
merged_df.to_csv(OUTPUT_COMPARISON, index=False)
print(f"\nFull comparison saved to: {OUTPUT_COMPARISON}")

# Save Disagreements (These are the most interesting ones to check!)
disagreements = merged_df[merged_df['All_Agree'] == False].sort_values(by='Avg_Confidence', ascending=False)
if len(disagreements) > 0:
    disagreements.to_csv(OUTPUT_DISAGREEMENTS, index=False)
    print(f"  {len(disagreements)} Disagreements saved to: {OUTPUT_DISAGREEMENTS}")
    print("   (Use these for your LLM Judge step!)")
else:
    print("The models agreed on everything (or something is wrong).")