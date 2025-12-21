import argparse
import os
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "models")

# Define the models to train
MODELS_TO_TRAIN = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "BioLinkBERT": "michiyasunaga/BioLinkBERT-base"
}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # Weighted metrics (Good for general performance)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    # Macro metrics (Crucial for minority classes in your project)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    
    acc = accuracy_score(labels, preds)
    
    return {
    "accuracy": acc,
    "f1_weighted": f1_w,
    "f1_macro": f1_m,
    "precision_weighted": precision_w,
    "precision_macro": precision_m,
    "recall_weighted": recall_w,
    "recall_macro": recall_m,
    }

def train_single_model(model_name, model_id, raw_dataset, epochs, lr):
    print(f"TRAINING: {model_name} ({model_id})")
    print()
    
    output_dir = os.path.join(RESULTS_DIR, model_name)
    
    print(f"[{model_name}] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)    
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    
    label_list = sorted(list(set(raw_dataset['train']['relation_type'])))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # Tokenize Data
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )
        tokenized["labels"] = [label2id[l] for l in examples["relation_type"]]
        return tokenized


    print(f"[{model_name}] Tokenizing...")
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

    # Initialize Data Collator for Dynamic Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. Load Model
    print(f"[{model_name}] Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    # Resize embedding layer to fit [E1] tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # 5. Training Args
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,

    learning_rate=lr,
    weight_decay=0.02,

    warmup_ratio=0.15,
    lr_scheduler_type="linear",

    logging_dir=os.path.join(output_dir, 'logs'),
    logging_steps=50,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,

    fp16=torch.cuda.is_available(),
    seed=42,

    report_to="none",
    push_to_hub=False,
    save_safetensors=False,
)

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    # Final Save
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Evaluate on Test Set
    print(f"[{model_name}] Evaluating on Test Set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    # Save text results
    with open(os.path.join(final_path, "test_results.txt"), "w") as f:
        for k, v in test_results.items():
            f.write(f"{k}: {v}\n")
            
    print(f"{model_name} Completed. Saved to {final_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning Rate")
    args = parser.parse_args()

    print("Loading Datasets from CSV...")
    data_files = {
        'train': os.path.join(DATA_DIR, "train.csv"),
        'validation': os.path.join(DATA_DIR, "dev.csv"),
        'test': os.path.join(DATA_DIR, "test.csv")
    }
    for k, v in data_files.items():
        if not os.path.exists(v):
            print(f"Error: {v} not found")
            return

    raw_dataset = load_dataset('csv', data_files=data_files)
    for name, model_id in MODELS_TO_TRAIN.items():
        try:
            train_single_model(name, model_id, raw_dataset, args.epochs, args.lr)
        except Exception as e:
            print(f"Failed to train {name}: {e}")

if __name__ == "__main__":
    main()