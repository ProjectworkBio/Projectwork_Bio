import os
import glob
import polars as pl
import spacy
from sentence_splitter import SentenceSplitter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import re
import warnings
import time
import logging
from multiprocessing import shared_memory
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", FutureWarning)

DATA_FOLDER = "Pubmed_abstracts_csvs"
OUT_FOLDER = "Result-v5"
# MODEL_NAME = "en_ner_jnlpba_md"
MODEL_NAME = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz"
PROTEIN_LIST = "protein_synonyms.csv"
SIM_THRESHOLD = 0.85
MIN_TOKEN_OVERLAP = 0.7
BATCH_SIZE = 500
N_WORKERS = None

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def cosine_safe(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.dot(a, b) / (na * nb) if na and nb else 0.0

def get_span_vector(span):
    vecs = [t.vector for t in span if t.has_vector and t.vector is not None and t.vector.size > 0]
    if len(vecs) == 0:
        return np.zeros(span.doc.vocab.vectors_length, dtype=np.float32)
    arr = np.asarray(vecs, dtype=np.float32)
    arr = arr.reshape(len(vecs), -1)
    return arr.mean(axis=0)

def token_overlap(candidate_tokens, sentence_tokens):
    return len(candidate_tokens & sentence_tokens) / max(len(candidate_tokens), 1)

def sentence_contains_protein(sent_tokens, protein_token_sets):
    for tokens in protein_token_sets.values():
        if tokens & sent_tokens:
            return True
    return False

# GLOBALS
nlp = None
splitter = None
protein_token_sets = None
protein_names = None
protein_map = None
emb_shm = None
emb_array_shape = None

# WORKER INITIALIZER
def worker_init(shared_mem_name, emb_shape, names_list, token_sets_dict, prot_map):
    global nlp, splitter, protein_token_sets, protein_names, protein_map, emb_shm, emb_array_shape

    nlp = spacy.load(MODEL_NAME, exclude=["parser", "tagger", "textcat", "lemmatizer", "tok2vec"])
    splitter = SentenceSplitter(language="en")

    # Attach to shared memory
    emb_shm = shared_memory.SharedMemory(name=shared_mem_name)
    emb_array_shape = emb_shape
    protein_names = names_list
    protein_token_sets = token_sets_dict
    protein_map = prot_map
    logger.info(f"Worker init done")

# Helper to get embedding vector from shared memory
def get_protein_vector(idx):
    return np.ndarray(emb_array_shape, dtype=np.float32, buffer=emb_shm.buf)[idx]

# PROCESS CSV
def process_csv_file(file_path):
    start_time = time.time()  # start timing
    df = pl.read_csv(file_path)
    results = []

    filtered_sentences = []
    filtered_pmids = []

    for row in df.iter_rows(named=True):
        pmid = row['PubMedID']
        abstract = row['AbstractText']
        if not abstract:
            continue
        sentences = splitter.split(abstract)
        for sent in sentences:
            sent_tokens = set(normalize(sent).split())
            if sentence_contains_protein(sent_tokens, protein_token_sets):
                filtered_sentences.append(sent)
                filtered_pmids.append(pmid)

    if not filtered_sentences:
        logger.info(f"No relevant sentences in {os.path.basename(file_path)}")
        return []

    docs = list(nlp.pipe(filtered_sentences, batch_size=BATCH_SIZE))

    for sent, pmid, doc in zip(filtered_sentences, filtered_pmids, docs):
        protein_entities = [e for e in doc.ents if e.label_ in {"GENE_OR_GENE_PRODUCT", "PROTEIN", "PROTEIN_COMPLEX"}]
        matched = set()
        sent_tokens = set(normalize(sent).split())

        for ent in protein_entities:
            vec = get_span_vector(ent)
            for idx, name in enumerate(protein_names):
                prot_vec = get_protein_vector(idx)
                if cosine_safe(vec, prot_vec) >= SIM_THRESHOLD and token_overlap(protein_token_sets[name], sent_tokens) >= MIN_TOKEN_OVERLAP:
                    matched.add(name)

        if matched:
            results.append({
                "PubMedID": pmid,
                "Sentence": sent,
                "Matched_Proteins": "; ".join(sorted(matched)),
                "UniProtIDs": "; ".join(sorted({protein_map[m] for m in matched}))
            })

    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        partial_file = os.path.join(OUT_FOLDER, os.path.basename(file_path).replace(".csv", "_matches.csv"))
        pl.DataFrame(results).write_csv(partial_file)
        elapsed = time.time() - start_time
        logger.info(f"Saved partial results: {partial_file} | Processed in {elapsed:.2f} seconds")

    return results

# MAIN
def main():
    pipeline_start = time.time()
    # Load proteins and compute embeddings once
    df = pl.read_csv(PROTEIN_LIST)
    protein_names = []
    protein_token_sets = {}
    protein_map = {}
    embeddings_list = []

    nlp_tmp = spacy.load(MODEL_NAME, exclude=["parser", "tagger", "textcat", "lemmatizer", "tok2vec"])

    for row in df.iter_rows(named=True):
        name = row['Protein']
        uniprot = row['UniProtID']
        protein_names.append(name)
        protein_map[name] = uniprot
        protein_token_sets[name] = set(normalize(name).split())

        doc = nlp_tmp(name)
        token_vecs = [t.vector for t in doc if t.has_vector and t.vector is not None and t.vector.size > 0]
        vec = np.mean(np.asarray(token_vecs, dtype=np.float32), axis=0) if token_vecs else np.zeros(nlp_tmp.vocab.vectors_length, dtype=np.float32)
        embeddings_list.append(vec)

    embeddings_array = np.vstack(embeddings_list).astype(np.float32)

    # Create shared memory for embeddings
    shm = shared_memory.SharedMemory(create=True, size=embeddings_array.nbytes)
    shared_array = np.ndarray(embeddings_array.shape, dtype=np.float32, buffer=shm.buf)
    np.copyto(shared_array, embeddings_array)

    # Launch ProcessPoolExecutor
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []

    initializer_args = (shm.name, embeddings_array.shape, protein_names, protein_token_sets, protein_map)

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=worker_init, initargs=initializer_args) as exe:
        futures = [exe.submit(process_csv_file, f) for f in all_files]
        for f in futures:
            try:
                rows = f.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Worker exception: {e}")

    # Save final CSV
    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_protein_matches.csv")
    if all_rows:
        pl.DataFrame(all_rows).write_csv(final_file)
    logger.info(f"DONE. Extracted {len(all_rows)} rows. Saved to {final_file}")
    total_elapsed = time.time() - pipeline_start
    logger.info(f"Total processing time: {total_elapsed:.2f} seconds")

    # Clean up shared memory
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    main()