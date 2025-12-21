import os
import glob
import polars as pl
import re
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModel

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

MODEL_ID = "kamalkraj/biobert_base_cased"
DATA_FOLDER = "Pubmed_abstracts_csvs"
OUT_FOLDER = "Result-BioBERT"
PROTEIN_LIST = "protein_synonyms.csv"
SIM_THRESHOLD = 0.7
N_WORKERS = 2  # threads

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def embed_text_tokens(text, tokenizer, model, device):
    """Return BioBERT embeddings for each token (not averaged)."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        # last_hidden_state shape: [1, seq_len, hidden_size]
        token_embeds = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]
    return token_embeds

tokenizer = None
model = None
protein_spans = None

def worker_init(prot_spans):
    global tokenizer, model, protein_spans
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(DEVICE)
    protein_spans = {k: v.to(DEVICE) for k, v in prot_spans.items()}
    logger.info("BioBERT GPU worker initialized.")


def process_csv_file(file_path):
    start_time = time.time()
    filename = os.path.basename(file_path)
    df = pl.read_csv(file_path)
    results = []

    for row in df.iter_rows(named=True):
        pmid = row["PubMedID"]
        abstract = row["AbstractText"]
        if not abstract:
            continue

        sentences = sent_tokenize(abstract)

        for sent in sentences:
            matched = []

            # get token-level embeddings for sentence
            sent_token_embeds = embed_text_tokens(sent, tokenizer, model, DEVICE)  # [sent_len, hidden_size]

            for prot_name, prot_token_embeds in protein_spans.items():
                sims = torch.nn.functional.cosine_similarity(
                    sent_token_embeds.unsqueeze(1),  # [sent_len, 1, hidden_size]
                    prot_token_embeds.unsqueeze(0),  # [1, prot_len, hidden_size]
                    dim=-1
                )  # [sent_len, prot_len]

                # average max similarity per protein token
                max_per_prot_token = sims.max(dim=0).values  # max over sentence tokens for each protein token
                mean_max_sim = max_per_prot_token.mean().item()

                if mean_max_sim >= SIM_THRESHOLD:
                    matched.append(prot_name)


            if matched:
                results.append({
                    "PubMedID": pmid,
                    "Matched_Proteins": "; ".join(list(dict.fromkeys(matched))),
                    "Relevant_Sentence": sent
                })

    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_file = os.path.join(OUT_FOLDER, f"{filename.replace('.csv','')}_partial.csv")
        pl.DataFrame(results).write_csv(out_file)
        logger.info(f"[INFO] Saved partial results: {out_file}")

    elapsed = time.time() - start_time
    logger.info(f"Processed {file_path} | {len(results)} matches | {elapsed:.2f}s")
    return results

if __name__ == "__main__":
    pipeline_start = time.time()

    df_prot = pl.read_csv(PROTEIN_LIST)
    protein_spans = {}

    tokenizer_ref = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model_ref = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(DEVICE)

    for row in df_prot.iter_rows(named=True):
        name = row["ProteinName"]
        text = name
        # store token-level embeddings
        protein_spans[name] = embed_text_tokens(text, tokenizer_ref, model_ref, DEVICE)

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []
    worker_init(protein_spans)
    process_csv_file(all_files[0])  
    # with ThreadPoolExecutor(max_workers=N_WORKERS, initializer=worker_init, initargs=(protein_spans,)) as exe:
    #     futures = [exe.submit(process_csv_file, f) for f in all_files]
    #     for f in futures:
    #         try:
    #             rows = f.result()
    #             if rows:
    #                 all_rows.extend(rows)
    #         except Exception as e:
    #             logger.error(f"Worker exception: {e}")

    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_protein_matches.csv")
    if all_rows:
        pl.DataFrame(all_rows).write_csv(final_file)
    logger.info(f"DONE. Extracted {len(all_rows)} rows. Saved to {final_file}")

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Total processing time: {total_elapsed:.2f} seconds")
