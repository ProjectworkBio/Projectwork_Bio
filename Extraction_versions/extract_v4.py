import os
import glob
import polars as pl
import spacy
from spacy.matcher import PhraseMatcher
from concurrent.futures import ProcessPoolExecutor
import re
import time
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", FutureWarning)

DATA_FOLDER = "Pubmed_abstracts_csvs"
OUT_FOLDER = "Result-v4"
NER_MODEL = "en_ner_jnlpba_md"
NLP_MODEL = "en_core_sci_lg"
PROTEIN_LIST = "protein_synonyms.csv"
SIM_THRESHOLD = 0.9
BATCH_SIZE = 500
N_WORKERS = 2

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    return re.sub(r"\s+", " ", text)

def token_overlap(protein_tokens: set, sent_tokens: set) -> int:
    return len(protein_tokens & sent_tokens)

nlp_ner = None
nlp_vec = None
protein_spans = None
protein_token_sets = None

def worker_init(prot_spans):
    global nlp_ner, nlp_vec, protein_spans

    nlp_ner = spacy.load("en_ner_jnlpba_md", exclude=["tagger", "lemmatizer", "attribute_ruler", "textcat"])
    nlp_vec = spacy.load("en_core_sci_lg", exclude=["ner", "tagger", "parser", "lemmatizer", "attribute_ruler", "textcat"])
    protein_spans = prot_spans

    logger.info("Worker initialized (NER + SciSpaCy vectors).")

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

        # Run NER on the full abstract
        doc = nlp_ner(abstract)

        # Process each sentence separately
        for sent in doc.sents:
            sent_doc = nlp_ner(sent.text)

            # Extract protein NER entities from this sentence
            protein_entities = [ent for ent in sent_doc.ents if ent.label_ == "PROTEIN"]
            if not protein_entities:
                continue  # skip sentences without proteins

            matched = []
            other = []

            # Step 2: similarity check in sci_lg vector space
            for ent in protein_entities:
                ent_vec = nlp_vec(ent.text)

                if not ent_vec.has_vector:
                    other.append(ent.text)
                    continue

                is_match = False
                for ref_name, ref_span in protein_spans.items():
                    if ref_span.has_vector:
                        sim = ent_vec.similarity(ref_span)
                        if sim >= SIM_THRESHOLD:
                            is_match = True
                            break

                if is_match:
                    matched.append(ent.text)
                else:
                    other.append(ent.text)

            if matched:
                matched = list(dict.fromkeys(matched))
                other = list(dict.fromkeys(other))

                results.append({
                    "PubMedID": pmid,
                    "Matched_Proteins": "; ".join(matched),
                    "Other_Proteins": "; ".join(other),
                    "Relevant_Sentence": sent.text
                })

    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_file = os.path.join(
            OUT_FOLDER, f"{filename.replace('.xml.gz', '').replace('.csv', '')}_partial.csv"
        )
        pl.DataFrame(results).write_csv(out_file)
        logger.info(f"[INFO] Saved partial results: {out_file}")

    elapsed = time.time() - start_time
    logger.info(f"Processed {file_path} | {len(results)} matches | {elapsed:.2f}s")

    return results

if __name__ == "__main__":
    pipeline_start = time.time()

    # Load proteins
    df_prot = pl.read_csv(PROTEIN_LIST)
    protein_spans = {}
    protein_token_sets = {}
    phrases = []

    nlp_tmp = spacy.load(NLP_MODEL, exclude=["tagger", "lemmatizer", "attribute_ruler", "textcat", "ner", "parser"])

    for row in df_prot.iter_rows(named=True):
        name = row['ProteinName']
        token_sources = [name]
        if row.get("ProteinSynonyms"):
            token_sources.extend(str(row["ProteinSynonyms"]).split(";"))
        if row.get("GeneName"):
            token_sources.append(str(row.get("GeneName", "")))
        if row.get("GeneSynonyms"):
            token_sources.extend(str(row.get("GeneSynonyms", "")).split(";"))

        phrases.extend(token_sources)
        protein_token_sets[name] = set(normalize(" ".join(token_sources)).split())
        protein_spans[name] = nlp_tmp(name)

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=worker_init, initargs=(protein_spans,)) as exe:
        futures = [exe.submit(process_csv_file, f) for f in all_files]
        for f in futures:
            try:
                rows = f.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Worker exception: {e}")

    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_protein_matches.csv")
    if all_rows:
        pl.DataFrame(all_rows).write_csv(final_file)
    logger.info(f"DONE. Extracted {len(all_rows)} rows. Saved to {final_file}")

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Total processing time: {total_elapsed:.2f} seconds")
