import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# --- CONFIGURATION ---
BASE_DIR = os.getcwd()

DATA_FILE = os.path.join(BASE_DIR, "data", "processed_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "predictions")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "BioBERT": os.path.join(MODEL_DIR, "BioBERT", "final_model"),
    "PubMedBERT": os.path.join(MODEL_DIR, "PubMedBERT", "final_model"),
    "BioLinkBERT": os.path.join(MODEL_DIR, "BioLinkBERT", "final_model"),
}

MAX_LENGTH = 256
BATCH_SIZE = 32

def load_unlabeled_dataset(csv_path):
    ds = load_dataset("csv", data_files={"data": csv_path})
    return ds["data"]

def tokenize_dataset(dataset, tokenizer):
    def preprocess(examples):
        return tokenizer(
            examples["marked_text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing unlabeled data"
    )

    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        cols.append("token_type_ids")

    tokenized.set_format(type="torch", columns=cols)
    return tokenized

# PREDICTION
def run_inference(model_name, model_path, dataset, device):
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    })

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    tokenized_ds = tokenize_dataset(dataset, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    loader = DataLoader(
        tokenized_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference [{model_name}]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            predictions.extend(preds.cpu().tolist())
            confidences.extend(probs.max(dim=-1).values.cpu().tolist())

    return [id2label[p] for p in predictions], confidences


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_unlabeled_dataset(DATA_FILE)

    base_df = pd.DataFrame({
        "PubMedID": dataset["PubMedID"],
        "Entity_1": dataset["Matched_Protein_Name"],
        "Entity_2": dataset["Other_protein"],
        "Sentence": dataset["Sentence"],
    })

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}")
            continue

        predicted_relations, confidence = run_inference(
            model_name, model_path, dataset, device
        )

        out_df = base_df.copy()
        out_df["Relation"] = predicted_relations
        out_df["Confidence"] = confidence

        out_file = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_predictions.csv"
        )
        out_df.to_csv(out_file, index=False)
        print(f"Saved predictions → {out_file}")

    print("\nAll models finished prediction.")

if __name__ == "__main__":
    main()
