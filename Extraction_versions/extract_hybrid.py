# Hybrid extraction pipeline with exact match -> NER -> protein pair extraction

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
OUT_FOLDER = "Result-hybrid"
NER_MODEL = "en_ner_jnlpba_md"
PROTEIN_LIST = "protein_synonyms.csv"
BATCH_SIZE = 500
N_WORKERS = 6

def normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]+", "", text))

nlp_ner = None
phrase_matcher = None
protein_lookup = None  # maps normalized protein string -> (protein_id, name)

def worker_init(_protein_lookup):
    global nlp_ner, matcher, protein_lookup
    nlp_ner = spacy.load(NER_MODEL, exclude=["tagger", "lemmatizer", "attribute_ruler", "textcat"])
    
    # Build PhraseMatcher for fast exact matches
    matcher = PhraseMatcher(nlp_ner.vocab, attr="LOWER")
    patterns = [nlp_ner.make_doc(name) for _, name in _protein_lookup.values()]
    matcher.add("PROTEINS", patterns)

    protein_lookup = _protein_lookup
    logger.info("Worker initialized (NER + PhraseMatcher).")

def process_csv_file(file_path):
    start_time = time.time()
    filename = os.path.basename(file_path)
    df = pl.read_csv(file_path, columns=["PubMedID", "AbstractText"])
    results = []

    for row in df.iter_rows(named=True):
        pmid = row["PubMedID"]
        abstract = row["AbstractText"]
        if not abstract:
            continue

        # Single NER pass for the whole abstract
        doc = nlp_ner(abstract)

        for sent in doc.sents:
            sent_text = sent.text
            sent_doc = sent 

            # Step 1: Exact match with PhraseMatcher
            matches = matcher(sent_doc)
            matched_from_list = set()
            for _, start, end in matches:
                span = sent_doc[start:end]
                norm_span = normalize(span.text)
                if norm_span in protein_lookup:
                    matched_from_list.add(norm_span)

            if not matched_from_list:
                continue

            # Step 2: Extract all PROTEIN entities in the sentence
            protein_mentions = [ent.text for ent in sent_doc.ents if ent.label_ == "PROTEIN"]
            protein_mentions = list(dict.fromkeys(protein_mentions))
            if len(protein_mentions) < 2:
                continue  # need at least a pair

            # Step 3: Produce pairs: (protein_from_list, other_protein)
            for matched_norm in matched_from_list:
                prot_id, prot_name = protein_lookup[matched_norm]
                for prot in protein_mentions:
                    if normalize(prot) == matched_norm:
                        continue  # skip identical

                    results.append({
                        "PubMedID": pmid,
                        "Matched_Protein_Id": f"{prot_id}",
                        "Matched_Protein_Name" : f"{prot_name}",
                        "Other_protein": prot,
                        "Sentence": sent_text,
                    })

    if results:
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_file = os.path.join(
            OUT_FOLDER, f"{filename.replace('.csv', '')}_partial.csv"
        )
        pl.DataFrame(results).write_csv(out_file)
        logger.info(f"Saved partial results: {out_file}")

    logger.info(f"Processed {file_path} | {len(results)} matches | {time.time() - start_time:.2f}s")
    return results


if __name__ == "__main__":
    pipeline_start = time.time()

    # Load protein list
    df_prot = pl.read_csv(PROTEIN_LIST, columns=["UniProtID", "ProteinName"])
    protein_lookup = {}
    for row in df_prot.iter_rows(named=True):
        prot_id = row.get("UniProtID", "NA")
        name = row["ProteinName"]
        norm = normalize(name)
        protein_lookup[norm] = (prot_id, name)

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_rows = []
 
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=worker_init,
        initargs=(protein_lookup,)
    ) as exe:
        futures = [exe.submit(process_csv_file, f) for f in all_files]
        for f in futures:
            try:
                rows = f.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Worker exception: {e}")

    # Save combined output
    os.makedirs(OUT_FOLDER, exist_ok=True)
    final_file = os.path.join(OUT_FOLDER, "results_protein_matches.csv")
    if all_rows:
        pl.DataFrame(all_rows).write_csv(final_file)

    logger.info(f"DONE. Extracted {len(all_rows)} rows. Saved to {final_file}")
    logger.info(f"Total processing time: {time.time() - pipeline_start:.2f} seconds")