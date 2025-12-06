import os
import gzip
import xml.etree.ElementTree as ET
import polars as pl
import spacy
from sentence_splitter import SentenceSplitter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import re
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)

# -----------------------------
# CONFIG
# -----------------------------
DATA_FOLDER = "Data"
OUT_FOLDER = "Result-v5"
MODEL_NAME = "en_ner_jnlpba_md"
PROTEIN_LIST = "proteins_original.txt"
SIM_THRESHOLD = 0.85
MIN_TOKEN_OVERLAP = 0.7
BATCH_SIZE = 500
N_WORKERS = None  # Default: os.cpu_count()

# -----------------------------
# HELPERS
# -----------------------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def cosine_safe(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.dot(a, b) / (na * nb) if na and nb else 0.0

def get_span_vector(span):
    vecs = [t.vector.get() for t in span if t.has_vector and t.vector is not None and t.vector.size > 0]
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

# -----------------------------
# WORKER INITIALIZER
# -----------------------------
nlp = None
splitter = None
protein_embeddings = None
protein_token_sets = None

def worker_init():
    global nlp, splitter, protein_embeddings, protein_token_sets

    # Use GPU if available
    spacy.require_gpu()
    
    print("Loading spacy model")
    # Load model with unnecessary components disabled
    nlp = spacy.load(MODEL_NAME, exclude=["parser", "tagger", "textcat", "lemmatizer", "tok2vec"])
    print("Model loaded")
    splitter = SentenceSplitter(language="en")

    # Load protein names and precompute token sets
    protein_embeddings = {}
    protein_token_sets = {}
    with open(PROTEIN_LIST, "r", encoding="utf8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            doc = nlp(name)
            token_vecs = [t.vector.get() for t in doc if t.has_vector and t.vector is not None and t.vector.size > 0]
            vec = np.mean(np.asarray(token_vecs, dtype=np.float32), axis=0) if token_vecs else np.zeros(nlp.vocab.vectors_length, dtype=np.float32)
            protein_embeddings[name] = vec
            protein_token_sets[name] = set(normalize(name).split())
    print("Worker init done")

# -----------------------------
# PROCESS FILE
# -----------------------------
def process_pubmed_file(file_path):
    timings = {}
    start_total = time.time()
    results = []
    filename = os.path.basename(file_path)

    # XML parse
    print("Opening gzip file")
    start = time.time()
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            root = tree.getroot()
    except Exception:
        print(f"[WARN] Skipping corrupted file: {filename}")
        return []
    timings['xml_parse'] = time.time() - start
    print("Loaded gzip file")

    # Collect sentences
    start = time.time()
    articles = root.findall(".//PubmedArticle")
    all_sentences = []
    pmid_map = []
    for article in articles:
        pmid = article.findtext(".//ArticleId[@IdType='pubmed']")
        if not pmid:
            continue
        abstracts = [abst.text for abst in article.findall(".//Abstract/AbstractText") if abst.text]
        if not abstracts:
            continue
        abstract_text = " ".join(abstracts)
        sentences = splitter.split(abstract_text)
        all_sentences.extend(sentences)
        pmid_map.extend([pmid] * len(sentences))
    timings['sentence_split'] = time.time() - start
    print("Sentence split done")

    if not all_sentences:
        return []

    # Pre-filter sentences
    start = time.time()
    filtered_sentences = []
    filtered_pmids = []
    for sent, pmid in zip(all_sentences, pmid_map):
        sent_tokens = set(normalize(sent).split())
        if sentence_contains_protein(sent_tokens, protein_token_sets):
            filtered_sentences.append(sent)
            filtered_pmids.append(pmid)
    timings['pre_filter'] = time.time() - start
    print("Pre-filter done")

    if not filtered_sentences:
        return []

    # SpaCy NER
    start = time.time()
    docs = list(nlp.pipe(filtered_sentences, batch_size=BATCH_SIZE))
    timings['spacy_time'] = time.time() - start
    print("SpaCy NER done")

    # Entity matching
    start = time.time()
    for sent, pmid, doc in zip(filtered_sentences, filtered_pmids, docs):
        protein_entities = [e for e in doc.ents if e.label_ in {"GENE_OR_GENE_PRODUCT", "PROTEIN", "PROTEIN_COMPLEX"}]
        matched = set()
        sent_tokens = set(normalize(sent).split())

        for ent in protein_entities:
            vec = get_span_vector(ent)
            for name, canon_vec in protein_embeddings.items():
                if cosine_safe(vec, canon_vec) >= SIM_THRESHOLD and token_overlap(protein_token_sets[name], sent_tokens) >= MIN_TOKEN_OVERLAP:
                    matched.add(name)

        if len(matched) >= 2:
            results.append({
                "PubMedID": pmid,
                "Matched_Proteins": "; ".join(sorted(matched)),
                "Relevant_Sentence": sent
            })
    timings['entity_processing'] = time.time() - start
    print("Entity matching done")

    # Save partial CSV
    start = time.time()
    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        partial_file = os.path.join(OUT_FOLDER, f"{filename.replace('.xml.gz','')}.csv")
        pl.DataFrame(results).write_csv(partial_file)
        print(f"[INFO] Saved partial results: {partial_file}")
    timings['write_csv'] = time.time() - start
    timings['total'] = time.time() - start_total

    print(f"[INFO] Finished {filename}: {len(results)} matches, Timings: {timings}")
    return results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    files = [
        os.path.join(DATA_FOLDER, f)
        for f in os.listdir(DATA_FOLDER)
        if f.endswith(".gz") and int(re.search(r"pubmed25n(\d+)", f).group(1)) > 802
    ]
    all_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=worker_init) as exe:
        futures = [exe.submit(process_pubmed_file, f) for f in files]
        for f in futures:
            try:
                rows = f.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                print(f"[ERROR] Worker exception: {e}")

    # Final CSV
    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_bioner.csv")
    pl.DataFrame(all_rows).write_csv(final_file)
    print(f"\nDONE. Extracted {len(all_rows)} rows. Saved to {final_file}")
