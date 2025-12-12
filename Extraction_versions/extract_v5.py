import os
import glob
import polars as pl
import spacy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import re
import time
import warnings

warnings.simplefilter("ignore", FutureWarning)

# ---------------------------------------
# CONFIG
# ---------------------------------------
DATA_FOLDER = "Pubmed_abstracts_csvs"
OUT_FOLDER = "Result-v5"
MODEL_NAME = "en_ner_jnlpba_md"
PROTEIN_LIST = "protein_synonyms.csv"
SIM_THRESHOLD = 0.85
MIN_TOKEN_OVERLAP = 0.7
BATCH_SIZE = 300
N_WORKERS = 2   # default: CPU count

# ---------------------------------------
# HELPERS
# ---------------------------------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def cosine_safe(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.dot(a, b) / (na * nb) if na and nb else 0.0

def get_span_vector(span):
    vecs = []
    for t in span:
        if t.has_vector and t.vector is not None and t.vector.size > 0:
            vecs.append(t.vector)
    if not vecs:
        return np.zeros(span.doc.vocab.vectors_length, dtype=np.float32)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr.mean(axis=0)

def token_overlap(a, b):
    return len(a & b) / max(len(a), 1)

# ---------------------------------------
# GLOBALS inside workers
# ---------------------------------------
nlp = None
protein_embeddings = None
protein_token_sets = None

def worker_init(protein_names):
    global nlp, protein_embeddings, protein_token_sets

    #spacy.require_gpu()

    print("[Worker] Loading NER model...")
    nlp = spacy.load(MODEL_NAME, exclude=["tagger", "lemmatizer", "textcat"])
    print("[Worker] NER model loaded.")

    protein_embeddings = {}
    protein_token_sets = {}

    # Precompute vectors + token sets
    for name in protein_names:
        doc = nlp(name)
        token_vecs = [
            t.vector for t in doc
            if t.has_vector and t.vector is not None and t.vector.size > 0
        ]

        if token_vecs:
            vec = np.mean(np.asarray(token_vecs, dtype=np.float32), axis=0)
        else:
            vec = np.zeros(nlp.vocab.vectors_length, dtype=np.float32)

        protein_embeddings[name] = vec
        protein_token_sets[name] = set(normalize(name).split())

    print("[Worker] Protein vectors ready.")

# ---------------------------------------
# PROCESS CSV FILE
# ---------------------------------------
def process_csv_file(file_path):
    start_t = time.time()
    filename = os.path.basename(file_path)

    df = pl.read_csv(file_path)
    results = []

    for row in df.iter_rows(named=True):
        pmid = row["PubMedID"]
        abstract = row["AbstractText"]
        if not abstract:
            continue

        # sentence splitting + NER backbone doc
        doc = nlp(abstract)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # prefilter by tokens BEFORE running full NER
            sent_tokens = set(normalize(sent_text).split())

            # skip sentences that have zero overlap with any protein tokens
            if not any(tset & sent_tokens for tset in protein_token_sets.values()):
                continue

            # run NER on the sentence
            sent_doc = nlp(sent_text)

            # extract entities
            ents = [e for e in sent_doc.ents if e.label_ in {"PROTEIN"}]
            if not ents:
                continue

            matched = set()

            # similarity matching
            for ent in ents:
                vec = get_span_vector(ent)

                for pname, pvec in protein_embeddings.items():
                    if cosine_safe(vec, pvec) >= SIM_THRESHOLD:
                        if token_overlap(protein_token_sets[pname], sent_tokens) >= MIN_TOKEN_OVERLAP:
                            matched.add(pname)

            # only save sentences with ≥ 2 matched proteins
            if len(matched) >= 2:
                results.append({
                    "PubMedID": pmid,
                    "Matched_Proteins": "; ".join(sorted(matched)),
                    "Relevant_Sentence": sent_text
                })


    # 5. Save partial result
    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_path = os.path.join(OUT_FOLDER, f"{filename.replace('.csv','')}_partial.csv")
        pl.DataFrame(results).write_csv(out_path)
        print(f"[INFO] Saved: {out_path}")

    print(f"[DONE] {file_path}: {len(results)} matches | {time.time()-start_t:.2f}s")
    return results

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":

    # Load protein name list
    df_prot = pl.read_csv(PROTEIN_LIST)
    protein_names = df_prot["ProteinName"].to_list()

    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=worker_init,
        initargs=(protein_names,)
    ) as exe:

        futures = [exe.submit(process_csv_file, f) for f in csv_files]
        for fut in futures:
            try:
                r = fut.result()
                if r:
                    all_rows.extend(r)
            except Exception as e:
                print(f"[ERROR] {e}")

    # Final merge
    os.makedirs(OUT_FOLDER, exist_ok=True)
    out_file = os.path.join(OUT_FOLDER, "final_results.csv")
    pl.DataFrame(all_rows).write_csv(out_file)

    print(f"\n[FINAL] Extracted {len(all_rows)} rows → {out_file}")
