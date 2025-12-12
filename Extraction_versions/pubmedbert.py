import os
import glob
import polars as pl
import re
import time
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", FutureWarning)

DATA_FOLDER = "Pubmed_abstracts_csvs"
OUT_FOLDER = "Result-PubMedBERT"
PROTEIN_LIST = "protein_synonyms.csv"

EMB_MODEL = "NeuML/pubmedbert-base-embeddings"
SIM_THRESHOLD = 0.7
N_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Helpers
# ------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def embed_tokens(text: str, tokenizer, model, device):
    """Return token-level embeddings (shape: [num_tokens, hidden_size])."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        out = model(**encoded).last_hidden_state  # [1, seq_len, hidden_size]
    return out.squeeze(0)  # [seq_len, hidden_size]

# ------------------------
# Worker globals
# ------------------------
embed_tokenizer = None
embed_model = None
protein_embeddings = None

# ------------------------
# Worker initialization
# ------------------------
def worker_init(prot_embs):
    global embed_tokenizer, embed_model, protein_embeddings
    logger.info("Initializing PubMedBERT worker...")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
    embed_model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
    embed_model.eval()
    protein_embeddings = prot_embs
    logger.info("Worker ready (PubMedBERT embeddings).")

# ------------------------
# CSV processing
# ------------------------
def process_csv_file(file_path):
    start = time.time()
    filename = os.path.basename(file_path)
    df = pl.read_csv(file_path)
    results = []

    for row in df.iter_rows(named=True):
        pmid = row["PubMedID"]
        abstract = row["AbstractText"]

        if not abstract or not isinstance(abstract, str):
            continue

        # Use nltk sentence tokenizer
        sentences = sent_tokenize(abstract)

        for sent in sentences:
            if not sent.strip():
                continue

            sent_token_embs = embed_tokens(sent, embed_tokenizer, embed_model, DEVICE)

            matched = []
            other = []

            for prot_name, prot_token_embs in protein_embeddings.items():
                # Compute cosine similarity across all token pairs
                sims = cosine_similarity(
                    sent_token_embs.unsqueeze(1),  # [sent_len, 1, hidden]
                    prot_token_embs.unsqueeze(0),  # [1, prot_len, hidden]
                    dim=-1
                )  # [sent_len, prot_len]

                max_sim = sims.max().item()
                if max_sim >= SIM_THRESHOLD:
                    matched.append(prot_name)
                else:
                    other.append(prot_name)

            if matched:
                results.append({
                    "PubMedID": pmid,
                    "Matched_Proteins": "; ".join(list(dict.fromkeys(matched))),
                    "Other_Proteins": "; ".join(list(dict.fromkeys(other))),
                    "Relevant_Sentence": sent
                })

    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_file = os.path.join(OUT_FOLDER, f"{filename.replace('.csv','')}_pubmedbert_partial.csv")
        pl.DataFrame(results).write_csv(out_file)
        logger.info(f"Saved partial results: {out_file}")

    elapsed = time.time() - start
    logger.info(f"Processed {file_path} | {len(results)} matches | {elapsed:.1f}s")
    return results

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    t0 = time.time()

    # Load protein list
    df_prot = pl.read_csv(PROTEIN_LIST)
    protein_embeddings = {}

    logger.info("Building protein embeddings (token-level) with PubMedBERT...")

    tokenizer_ref = AutoTokenizer.from_pretrained(EMB_MODEL)
    model_ref = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
    model_ref.eval()

    for row in df_prot.iter_rows(named=True):
        name = row["ProteinName"]
        # Only use canonical protein name (avoid noisy short synonyms)
        prot_token_embs = embed_tokens(name, tokenizer_ref, model_ref, DEVICE)
        protein_embeddings[name] = prot_token_embs

    del tokenizer_ref
    del model_ref
    logger.info("Protein embeddings complete.")

    worker_init(protein_embeddings)
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []

    for f in all_files:
        rows = process_csv_file(f)
        if rows:
            all_rows.extend(rows)

    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_pubmedbert.csv")
    if all_rows:
        pl.DataFrame(all_rows).write_csv(final_file)

    logger.info(f"DONE. Extracted {len(all_rows)} rows. Saved to {final_file}")
    logger.info(f"Total time: {time.time() - t0:.1f} seconds")
