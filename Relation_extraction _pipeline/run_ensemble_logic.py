import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Pred_DIR  = os.path.join(BASE_DIR, "results", "predictions")
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")
CONF_THRESHOLD = 0.80

os.makedirs(ANALYSIS_DIR, exist_ok=True)

MODEL_RESULTS = {
    "BioBERT":     os.path.join(Pred_DIR, "BioBERT_predictions.csv"),
    "PubMedBERT":  os.path.join(Pred_DIR, "PubMedBERT_predictions.csv"),
    "BioLinkBERT": os.path.join(Pred_DIR, "BioLinkBERT_predictions.csv"),
}

JOIN_KEYS = ["PubMedID", "Entity_1", "Entity_2", "Sentence"]

FULL_COMPARISON_OUT = os.path.join(ANALYSIS_DIR, "full_model_comparison.csv")
FINAL_ENSEMBLE_OUT  = os.path.join(ANALYSIS_DIR, "final_ensemble_report.csv")
HARD_CASES_OUT      = os.path.join(ANALYSIS_DIR, "hard_disagreements_for_llm_judge.csv")

MODELS = list(MODEL_RESULTS.keys())

def load_and_merge_results():
    merged_df = None
    print("=== Loading and merging model predictions ===")

    for model, path in MODEL_RESULTS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prediction file: {path}")

        df = pd.read_csv(path)
        df = df[JOIN_KEYS + ["Relation", "Confidence"]].copy()

        df.rename(columns={
            "Relation": f"{model}_Relation",
            "Confidence": f"{model}_Conf"
        }, inplace=True)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=JOIN_KEYS, how="inner")

    print(f"Aligned {len(merged_df)} samples across all models.")
    return merged_df

def analyze_agreement(df):
    rel_cols  = [f"{m}_Relation" for m in MODELS]
    conf_cols = [f"{m}_Conf" for m in MODELS]

    df["All_Agree"] = df[rel_cols].nunique(axis=1) == 1
    df["Avg_Confidence"] = df[conf_cols].mean(axis=1)

    return df


def ensemble_decision(row):
    confident_preds = []

    for model in MODELS:
        rel = row[f"{model}_Relation"]
        conf = row[f"{model}_Conf"]

        if pd.notna(rel) and conf >= CONF_THRESHOLD:
            confident_preds.append((model, rel, conf))

    # Case (i): |C_tau| = 0
    if len(confident_preds) == 0:
        return None, None, "Escalate_No_Confident_Model"

    # Case (ii): |C_tau| = 1
    if len(confident_preds) == 1:
        return None, None, "Escalate_Single_Model"

    # Case (iii): |C_tau| >= 2 but no majority
    labels = [p[1] for p in confident_preds]
    vote_counts = Counter(labels)
    top_label, top_count = vote_counts.most_common(1)[0]

    if top_count <= len(confident_preds) // 2:
        return None, None, "Escalate_No_Majority"

    # Majority agreement → accept ensemble decision
    best_conf = max(
        p[2] for p in confident_preds if p[1] == top_label
    )

    return top_label, best_conf, "Accepted_By_Ensemble"


def main():
    df = load_and_merge_results()
    df = analyze_agreement(df)

    print("\n=== Running Ensemble Voting ===")
    tqdm.pandas(desc="Ensembling")
    df[["Ensemble_Result", "Ensemble_Conf", "Decision_Method"]] = df.progress_apply(
        lambda r: pd.Series(ensemble_decision(r)), axis=1
    )

    # Save full comparison
    df.to_csv(FULL_COMPARISON_OUT, index=False)
    print(f"Full comparison saved → {FULL_COMPARISON_OUT}")

    # Save final ensemble view
    results = df.apply(
    lambda r: pd.Series(ensemble_decision(r),
                        index=["Ensemble_Relation", "Ensemble_Confidence", "Decision_Status"]),
    axis=1
)

    df = pd.concat([df, results], axis=1)
    accepted = df[df["Decision_Status"] == "Accepted_By_Ensemble"]
    escalated = df[df["Decision_Status"] != "Accepted_By_Ensemble"]
    accepted_cols = [
        "PubMedID", "Entity_1", "Entity_2", "Sentence",
        "BioBERT_Relation", "PubMedBERT_Relation", "BioLinkBERT_Relation"
    ]

    accepted.to_csv(FINAL_ENSEMBLE_OUT, index=False)
    escalated[accepted_cols].to_csv(HARD_CASES_OUT, index=False)

    print("\n=== Ensemble Summary ===")
    print(f"Total instances:        {len(df)}")
    print(f"Accepted by ensemble:   {len(accepted)}")
    print(f"Escalated to LLM Judge: {len(escalated)}\n")

    print("Escalation breakdown:")
    print(escalated["Decision_Status"].value_counts())

    print(f"\nSaved ensemble output → {FINAL_ENSEMBLE_OUT}")
    print(f"Saved LLM hard cases   → {HARD_CASES_OUT}")


if __name__ == "__main__":
    main()
