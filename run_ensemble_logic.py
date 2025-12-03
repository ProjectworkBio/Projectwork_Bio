import pandas as pd
import os
import numpy as np
from collections import Counter

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'results', 'analysis', 'full_model_comparison_v2.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'results', 'analysis', 'final_ensemble_report_v2.csv')

# Models involved in the vote
MODELS = ["BioBERT", "PubMedBERT", "BioLinkBERT"]

def get_ensemble_prediction(row):
    """
    Determines the final label based on Majority Vote + Confidence for Tie-Breaking.
    """
    votes = []
    confidences = {}
    
    for model in MODELS:
        pred_col = f"{model}_Relation"
        conf_col = f"{model}_Conf"
        
        if pred_col in row and pd.notna(row[pred_col]):
            pred = row[pred_col]
            conf = row[conf_col]
            votes.append(pred)
            if pred not in confidences or conf > confidences[pred]:
                confidences[pred] = conf

    if not votes: return "NO_RELATION", 0.0

    # Majority Vote Logic
    vote_counts = Counter(votes)
    top_prediction, count = vote_counts.most_common(1)[0]
    
    # Case A: Majority Vote
    if count > 1:
        method = "Majority_Vote"
        final_label = top_prediction
        final_conf = confidences[final_label] 
        
    # Case B: 3-Way Tie so Use Highest Confidence
    else:
        method = "Confidence_TieBreak"
        final_label = max(confidences, key=confidences.get)
        final_conf = confidences[final_label]

    return final_label, final_conf, method

def main():
    print("--- Running Ensemble Logic ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found: {INPUT_CSV}")
        print("Please run 'analyze_model_agreement.py' first!")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} comparisons.")

    # Apply Ensemble Logic
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Voting")
        df[['Ensemble_Result', 'Ensemble_Conf', 'Decision_Method']] = df.progress_apply(
            lambda row: pd.Series(get_ensemble_prediction(row)), axis=1
        )
    except ImportError:
        df[['Ensemble_Result', 'Ensemble_Conf', 'Decision_Method']] = df.apply(
            lambda row: pd.Series(get_ensemble_prediction(row)), axis=1
        )

    # SAVE RESULTS
    cols = ['PubMedID', 'Ensemble_Result', 'Ensemble_Conf', 'Decision_Method', 
            'Entity_1_Raw', 'Entity_2_Raw', 'Sentence'] + \
           [c for c in df.columns if c not in ['PubMedID', 'Ensemble_Result', 'Ensemble_Conf', 'Decision_Method', 'Entity_1_Raw', 'Entity_2_Raw', 'Sentence']]
    
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nEnsemble Statistics:")
    print(df['Decision_Method'].value_counts())
    
    print(f"\nFinal Ensemble Report saved to: {OUTPUT_CSV}")
    
    # Identify pure disagreements for LLM Judge
    disagreements = df[df['Decision_Method'] == "Confidence_TieBreak"]
    print(f"Found {len(disagreements)} complex disagreements (3-way split).")
    if len(disagreements) > 0:
        disagreements.to_csv(os.path.join(BASE_DIR, 'results', 'analysis', 'hard_disagreements_for_llm.csv'), index=False)
        print(f"   Saved specific disagreements to 'hard_disagreements_for_llm.csv' - Use BioMistral on these!")

if __name__ == "__main__":
    main()