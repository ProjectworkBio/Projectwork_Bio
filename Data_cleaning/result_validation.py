import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Load protein synonyms dictionary
prot = pd.read_csv("protein_synonyms.csv")

def split_syn(x):
    if pd.isna(x):
        return []
    return [s.strip().lower() for s in x.split(",")]

# Build dictionary: synonym -> UniProtID
syn_to_uniprot = {}

for _, row in prot.iterrows():
    uid = row["UniProtID"]
    
    # collect synonyms from ProteinSynonyms and GeneSynonyms + primary names
    syns = []
    syns += split_syn(row.get("ProteinSynonyms"))
    syns += split_syn(row.get("GeneSynonyms"))
    syns.append(str(row.get("ProteinName", "")).lower())
    syns.append(str(row.get("GeneName", "")).lower())
    
    # map
    for s in syns:
        if len(s) >= 1:     # allow all lengths; adjust if needed
            syn_to_uniprot[s] = uid

# Load extracted matches
results = pd.read_csv("Result-v5\\results_protein_matches.csv")

def split_multi(x):
    if pd.isna(x):
        return []
    return [s.strip().lower() for s in x.split(";")]

# Output columns
tp_list = []
fp_list = []

rows_eval = []

# Evaluate each match
for _, row in results.iterrows():
    pub = row["PubMedID"]
    sent = row["Sentence"]
    matched_proteins = split_multi(row["Matched_Proteins"])
    matched_uniprot = split_multi(row["UniProtIDs"])
    
    for mp, uid in zip(matched_proteins, matched_uniprot):
        
        # Check if the synonym exists and maps to the same UniProt ID
        if mp in syn_to_uniprot and syn_to_uniprot[mp].lower() == uid.lower():
            tp_list.append(1)
            fp_list.append(0)
            label = "TP"
        else:
            tp_list.append(0)
            fp_list.append(1)
            label = "FP"
        
        rows_eval.append({
            "PubMedID": pub,
            "Matched_Synonym": mp,
            "Predicted_UniProt": uid,
            "True_UniProt": syn_to_uniprot.get(mp, None),
            "Label": label,
            "Sentence": sent
        })

eval_df = pd.DataFrame(rows_eval)
eval_df.to_csv("evaluation_output.csv", index=False)


# 4. Confusion matrix & metrics
y_true = [1 if r["Label"] == "TP" else 0 for _, r in eval_df.iterrows()]
y_pred = [1] * len(y_true)   # every match is predicted positive

cm = confusion_matrix(y_true, y_pred, labels=[1,0])
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, pos_label=1, average="binary"
)

print("\nConfusion Matrix")
print("            Pred=Pos")
print(f"True Pos      {cm[0][0]}")
print(f"False Pos     {cm[1][0]}")

print("\nMetrics")
print(f"Precision: {precision:.4f}")

print("\nSaved detailed evaluation to evaluation_output.csv")
