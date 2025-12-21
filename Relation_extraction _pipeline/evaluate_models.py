import os
import json
import time
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for d in [PREDICTIONS_DIR, METRICS_DIR, ]:
    os.makedirs(d, exist_ok=True)

TEST_FILES = [
    os.path.join(DATA_DIR, "test.csv"),
]

MODELS = {
    "BioBERT": os.path.join(MODEL_DIR, "BioBERT", "final_model"),
    "PubMedBERT": os.path.join(MODEL_DIR, "PubMedBERT", "final_model"),
    "BioLinkBERT": os.path.join(MODEL_DIR, "BioLinkBERT" , "final_model"),
}

MAX_LENGTH = 256
BATCH_SIZE = 32

def load_test_dataset(csv_path):
    ds = load_dataset("csv", data_files={"test": csv_path})
    return ds["test"]

def preprocess_dataset(dataset, tokenizer):
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing test set"
    )

    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        cols.append("token_type_ids")

    tokenized.set_format(type="torch", columns=cols)
    return tokenized

# MODEL EVALUATION
def evaluate_model(model_name, model_path, test_dataset, device):
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    tokenized_test = preprocess_dataset(test_dataset, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, collate_fn=collator)

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference [{model_name}]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(probs.max(dim=-1).values.cpu().tolist())

    pred_labels = [id2label[p] for p in all_preds]
    return pred_labels, all_confs, id2label

# MAIN EVALUATION
def evaluate_on_file(csv_path, device):
    print(f"\n=== Evaluating test file: {csv_path} ===")
    test_ds = load_test_dataset(csv_path)

    y_true = test_ds["relation_type"]
    texts = test_ds["text"]
    e1s = test_ds["entity_1"]
    e2s = test_ds["entity_2"]

    all_metrics = []
    all_predictions = []

    for model_name, model_path in MODELS.items():
        y_pred, confs, id2label = evaluate_model(
            model_name, model_path, test_ds, device
        )

        labels = [id2label[i] for i in sorted(id2label.keys())]

        # overall averages: micro, macro, weighted
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        acc = accuracy_score(y_true, y_pred)

        all_metrics.append({
            "test_file": os.path.basename(csv_path),
            "model": model_name,
            "accuracy": round(acc, 4),
            "micro_precision": round(float(micro_p), 4),
            "micro_recall": round(float(micro_r), 4),
            "micro_f1": round(float(micro_f1), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        })

        for i in range(len(y_pred)):
            all_predictions.append({
                "text": texts[i],
                "entity_1": e1s[i],
                "entity_2": e2s[i],
                "gold_relation": y_true[i],
                "predicted_relation": y_pred[i],
                "confidence": round(confs[i], 4),
                "model": model_name,
                "test_file": os.path.basename(csv_path),
            })

    return pd.DataFrame(all_metrics), pd.DataFrame(all_predictions)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_metrics = []
    all_preds = []

    for csv_path in TEST_FILES:
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing test file: {csv_path}")
            continue

        metrics_df, preds_df = evaluate_on_file(csv_path, device)

        all_metrics.append(metrics_df)
        all_preds.append(preds_df)

        metrics_df.to_csv(
            os.path.join(METRICS_DIR, f"metrics_{os.path.basename(csv_path)}"),
            index=False
        )
        preds_df.to_csv(
            os.path.join(PREDICTIONS_DIR, f"predictions_{os.path.basename(csv_path)}"),
            index=False
        )

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary.to_csv(
            os.path.join(METRICS_DIR, "metrics_summary.csv"),
            index=False
        )

    if all_preds:
        preds_all = pd.concat(all_preds, ignore_index=True)
        preds_all.to_csv(
            os.path.join(PREDICTIONS_DIR, "predictions_all_models.csv"),
            index=False
        )

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
