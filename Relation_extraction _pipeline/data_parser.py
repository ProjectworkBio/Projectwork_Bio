import pandas as pd
import os
import re
import spacy
from tqdm import tqdm
from sklearn.utils import resample
import itertools
from setup_data import setup

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
REGULATOME_DIR = os.path.join(DATA_DIR, 'RegulaTome-corpus')
BIORED_DIR = os.path.join(DATA_DIR, 'BioRED', 'BioRED')
BIORED_FILES = {
    'train': 'Train.PubTator',
    'dev': 'Dev.PubTator',
    'test': 'Test.PubTator'
}
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_FILE = os.path.join(OUTPUT_DIR, 'prep_report.txt')

# --- BALANCING HYPERPARAMETERS (TRAINING ONLY) ---
PRUNE_THRESHOLD = 20      # Drop classes with < 20 samples 
NEGATIVE_RATIO = 4.0      # n Negatives for every 1 Positive (1:n ratio)

# --- UNIFIED MAPPING ---
UNIFIED_MAPPING = {
    # Regulation (Merge RegulaTome + BioRED Similar Relations)
    "Positive_regulation": "Positive_regulation",
    "Positive_Correlation": "Positive_regulation",
    "Negative_regulation": "Negative_regulation",
    "Negative_Correlation": "Negative_regulation",    
    "Out-of-scope" :"NO_RELATION",
    "Comparison": "NO_RELATION",
    "Cotreatment" : "NO_RELATION",
}

def log_message(message, file_handle):
    print(message)
    file_handle.write(str(message) + '\n')

def insert_entity_markers(sentence, e1, e2):
    if e1 not in sentence or e2 not in sentence:
        return None

    sentence = sentence.replace(e1, f"[E1]{e1}[/E1]", 1)
    sentence = sentence.replace(e2, f"[E2]{e2}[/E2]", 1)
    return sentence
import re

# --- PARSING FUNCTIONS ---
# Parses BRAT files from RegulaTome
def parse_regulatome(split_dir, nlp):
    if not os.path.exists(split_dir): 
        return pd.DataFrame()
    
    parsed = []

    for ann_file in os.listdir(split_dir):
        if not ann_file.endswith('.ann'): 
            continue
        
        file_id = ann_file.replace(".ann", "")
        txt_path = os.path.join(split_dir, file_id + '.txt')
        ann_path = os.path.join(split_dir, ann_file)
        
        try:
            with open(txt_path, encoding='utf-8') as f: 
                full_text = f.read()
            doc = nlp(full_text)

            entities = {}
            relations = []
            
            with open(ann_path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('T'):
                        tid, span, mention = line.strip().split("\t")  # example : ['T1', 'Protein 0 5', 'Atg21']
                        start, end = map(int, span.split()[1:3])
                        entities[tid] = {"text": mention, "start": start, "end": end}
                    elif line.startswith('R'):
                        rid, info = line.strip().split("\t")  # example : ['R1', 'Regulation Arg1:T1 Arg2:T2']
                        rel, a1, a2 = info.split()
                        relations.append((rel, a1.split(":")[1], a2.split(":")[1]))  
            for sent in doc.sents:
                sent_ents = {
                    tid: ent for tid, ent in entities.items()
                    if ent["start"] >= sent.start_char and ent["end"] <= sent.end_char
                }
                pos_pairs = set()
                for rel, e1, e2 in relations:
                    if e1 in sent_ents and e2 in sent_ents:
                        entity_1 = sent_ents[e1]["text"]
                        entity_2 = sent_ents[e2]["text"]
                        sen_marked = insert_entity_markers(sent.text, entity_1, entity_2)
                        parsed.append({
                            "text": sen_marked,
                            "entity_1": entity_1,
                            "entity_2": entity_2,
                            "relation_type": rel
                        })
                        pos_pairs.add(tuple(sorted((e1, e2))))
                
                for id1, id2 in itertools.combinations(sent_ents.keys(), 2):
                    if (tuple(sorted((id1, id2))) not in pos_pairs) and (id1 in sent_ents and id2 in sent_ents):
                        entity_1 = sent_ents[id1]["text"]
                        entity_2 = sent_ents[id2]["text"]
                        sen_marked = insert_entity_markers(sent.text, entity_1, entity_2)
                        parsed.append({
                            "text": sen_marked,
                            "entity_1": entity_1,
                            "entity_2": entity_2,
                            "relation_type": "NO_RELATION"
                        })
                 
        except Exception as e:
            print(f"[RegulaTome parse error] {file_id}: {e}")
    return pd.DataFrame(parsed)



# Parses PubTator file from BioRED
def parse_biored(file_path, nlp):
    if not os.path.exists(file_path): 
        return pd.DataFrame()
    
    with open(file_path, encoding='utf-8') as f: 
        content = f.read().strip()
    documents = content.split('\n\n')
    parsed = []
    
    for doc in documents:
        lines = doc.split('\n')
        if not lines: 
            continue
        
        title, abstract = "", ""
        entities = {}
        relations = []
        
        for line in lines:
            if '|t|' in line: 
                title = line.split('|t|')[1]
            elif '|a|' in line: 
                abstract = line.split('|a|')[1]
            elif '\t' in line:
                parts = line.split('\t')
                # Entity
                if len(parts) == 6:   # example : [14510914, 55, 78, sodium/iodide symporter, GeneOrGeneProduct, 6528]  
                    start, end = int(parts[1]), int(parts[2])
                    entities[parts[5]] = {'text': parts[3], 'start': start, 'end': end}
                # Relation
                elif len(parts) == 5: # example : [14510914, Association, D050033, D007454, No]
                    relations.append((parts[1], parts[2], parts[3]))

        full_text = f"{title} {abstract}"
        doc = nlp(full_text)
        for sent in doc.sents:
                sent_ents = {
                    tid: ent for tid, ent in entities.items()
                    if ent["start"] >= sent.start_char and ent["end"] <= sent.end_char
                }

                for rel, e1, e2 in relations:
                    if e1 in sent_ents and e2 in sent_ents:
                        entity_1 = sent_ents[e1]["text"]
                        entity_2 = sent_ents[e2]["text"]
                        sen_marked = insert_entity_markers(sent.text, entity_1, entity_2)
                        parsed.append({
                            "text": sen_marked,
                            "entity_1": entity_1,
                            "entity_2": entity_2,
                            "relation_type": rel        
                        })           
    return pd.DataFrame(parsed)


# --- PROCESSING PIPELINE ---
def process_split(split_name, log_f, nlp=None, valid_classes=None):
    log_message(f"\n{'='*30}\nProcessing {split_name.upper()} Split\n{'='*30}", log_f)

    # Parse RegulaTome and BioRED (Train/Dev/Test specific folder)
    rt_folder_name = 'devel' if split_name == 'dev' else split_name
    df_reg = parse_regulatome(os.path.join(REGULATOME_DIR, rt_folder_name), nlp)
    log_message(f"RegulaTome: {len(df_reg)} rows", log_f)
    df_bio = parse_biored(os.path.join(BIORED_DIR, BIORED_FILES[split_name]), nlp)
    log_message(f"BioRED: {len(df_bio)} rows", log_f)

    # Combine & Map
    df_combined = pd.concat([df_reg, df_bio], ignore_index=True)
    # Map the specified classes, leave others as it is
    df_combined['relation_type'] = df_combined['relation_type'].map(UNIFIED_MAPPING).fillna(df_combined['relation_type'])
    # Clean data
    df_combined = df_combined.dropna(subset=['relation_type'])
    # Balancing Logic (TRAIN ONLY)
    if split_name == 'train':
        log_message("--- Balancing Training Data ---", log_f)
        df_pos = df_combined[df_combined['relation_type'] != "NO_RELATION"]
        df_neg = df_combined[df_combined['relation_type'] == "NO_RELATION"]
        balanced_pos_dfs = []
        for relation, group in df_pos.groupby('relation_type'):
            count = len(group)
            if count < PRUNE_THRESHOLD:
                log_message(f"  - Dropping '{relation}' (n={count} < {PRUNE_THRESHOLD})", log_f)
                continue
            balanced_pos_dfs.append(group)
        if not balanced_pos_dfs:
            log_message("No positive classes survived pruning!", log_f)
            return set() 

        df_pos_final = pd.concat(balanced_pos_dfs) 
        valid_training_classes = set(df_pos_final['relation_type'].unique())
        valid_training_classes.add('NO_RELATION')
        n_pos = len(df_pos_final)
        n_neg = int(n_pos * NEGATIVE_RATIO)
        if len(df_neg) > n_neg:
            log_message(f"  - Downsampling Negatives: {len(df_neg)} -> {n_neg} (Ratio 1:{NEGATIVE_RATIO})", log_f)
            df_neg_final = df_neg.sample(n=n_neg, random_state=42)
        else:
            df_neg_final = df_neg
        df_final = pd.concat([df_pos_final, df_neg_final])
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        df_final.to_csv(out_path, index=False)
        
        log_message(f"Saved: {out_path} ({len(df_final)} samples)", log_f)
        if not df_final.empty:
            log_message("Classes Distribution:", log_f)
            log_message(df_final['relation_type'].value_counts().to_string(), log_f)   
        # Return the set of classes we kept so we can filter dev/test
        return valid_training_classes
        
    else:
        # For Dev/Test: No Balancing, but we MUST filter data to match Training classes
        if valid_classes is not None:
            # Filter to keep only classes seen in training
            initial_count = len(df_combined)
            df_final = df_combined[df_combined['relation_type'].isin(valid_classes)]
            filtered_count = len(df_final)
            dropped = initial_count - filtered_count          
            if dropped > 0:
                log_message(f"  - Filtered {dropped} rows containing classes not in Training set.", log_f)
        else:
            df_final = df_combined
            log_message("  - Warning: No valid_classes provided for filtering.", log_f)
    
        # Shuffle & Save
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        df_final.to_csv(out_path, index=False)
            
        log_message(f"Saved: {out_path} ({len(df_final)} samples)", log_f)
        if not df_final.empty:
            log_message("Classes Distribution:", log_f)
            log_message(df_final['relation_type'].value_counts().to_string(), log_f)        
        return set()


def main():
    setup()
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    try: nlp = spacy.load("en_core_web_sm")
    except: 
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    with open(LOG_FILE, 'w') as log_f:
        log_message("=== DATASET PREPARATION REPORT ===", log_f)
        valid_classes = process_split('train', log_f, nlp)     
        process_split('dev', log_f, nlp, valid_classes)
        process_split('test', log_f, nlp, valid_classes)

if __name__ == "__main__":
    main()

