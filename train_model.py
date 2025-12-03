import pandas as pd
import os
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# CONFIGURATION 
# BioBert: dmis-lab/biobert-base-cased-v1.2 or dmis-lab/biobert-base-cased-v1.1 
# PubMedBERT: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# BioLinkBERT: "michiyasunaga/BioLinkBERT-base"
# SciBert: allenai/scibert_scivocab_uncased
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "processed_data"
MODEL_OUTPUT_DIR = "model_output/BioBERT"
TRAIN_FILE = os.path.join(BASE_DIR ,DATA_DIR, "train_balanced.csv")
DEV_FILE = os.path.join(BASE_DIR, DATA_DIR, "dev_processed.csv")
TEST_FILE = os.path.join(BASE_DIR, DATA_DIR, "test_processed.csv")

# LOAD DATASET
print("Loading datasets...")
# Load the CSVs into a Hugging Face DatasetDict
dataset = load_dataset('csv', data_files={
    'train': TRAIN_FILE,
    'validation': DEV_FILE,
    'test': TEST_FILE
})

# CREATE LABEL MAPPINGS
print("Creating label mappings...")
# Get the unique list of labels from the training set
label_list = dataset['train'].unique('relation_type')
label_list.sort() 
num_labels = len(label_list)

# Two-way mapping
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Found {num_labels} unique labels: {label_list}")

# TOKENIZATION and PREPROCESSING
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """
    Tokenizes the input. The sentence is the main text, and the two entities are concatenated and used as the 'text_pair'.
    Might use Entity Markers is later updates.
    This gives the model context for both the sentence and the specific entities.
    """
    text_pair = [e1 + " " + tokenizer.sep_token + " " + e2 for e1, e2 in zip(examples['entity_1'], examples['entity_2'])]
    
    tokenized_inputs = tokenizer(
        examples['sentence'], 
        text_pair, 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )
    
    tokenized_inputs["labels"] = [label2id[label] for label in examples['relation_type']]
    return tokenized_inputs

print("Tokenizing datasets... (This may take a few minutes)")
tokenized_datasets = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

def compute_metrics(pred):
    """
    Calculates precision, recall, F1, and accuracy.
    This is called by the Trainer during evaluation.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # We use 'weighted' to account for class imbalance in the test set
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# LOAD PRE-TRAINED MODEL
print("Loading pre-trained BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# TRAINER
print("Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    
    # Evaluation & Saving Strategy
    eval_strategy="epoch", 
    save_strategy="epoch",
    
    # Best Model Logic
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True, 
    save_total_limit=2,     
    
    push_to_hub=False,
    save_safetensors=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# TRAIN THE MODEL
print("\n--- STARTING MODEL TRAINING ---")
trainer.train()
print("--- TRAINING COMPLETE ---")

# FINAL EVALUATION ON TEST SET
print("\n--- EVALUATING ON TEST SET ---")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

print("\n--- TEST SET RESULTS ---")
print(test_results)

# Save the final results to a file
with open(os.path.join(MODEL_OUTPUT_DIR, "final_test_results.txt"), "w") as f:
    f.write("--- FINAL TEST SET RESULTS ---\n")
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")

print(f"\nTraining complete. Model and results saved to '{MODEL_OUTPUT_DIR}'")