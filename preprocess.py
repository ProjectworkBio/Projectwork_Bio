import os
import re
import pandas as pd
import itertools
import shutil
import sys

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join('.', 'RegulaTome-corpus')
OUTPUT_DIR = 'processed_data'
LOG_FILE = os.path.join(BASE_DIR, OUTPUT_DIR, 'data_preparation_report.txt')
MIN_FREQ_THRESHOLD = 20  
NEG_POS_RATIO = 1.0      # 1:1 ratio of negative to positive samples
RANDOM_STATE = 42

# Re-label all of these relations that are not real as NO_RELATION
EXCLUDE_LIST = [
    'Out-of-scope', 'Equiv', 'Negated', 'Other', 'NO_RELATION'
]

def log_message(message, file_handle):
    """Prints a message to the console and appends it to the log file."""
    print(message)
    file_handle.write(str(message) + '\n')

def parse_brat_directory(directory_path: str) -> pd.DataFrame:
    """Parses a single BRAT directory into a DataFrame."""
    parsed_data = []
    
    arg1_re = re.compile(r'Arg1:(T\d+)')
    arg2_re = re.compile(r'Arg2:(T\d+)')

    ann_files = [f for f in os.listdir(directory_path) if f.endswith('.ann')]
    
    for ann_file in ann_files:
        file_id = os.path.splitext(ann_file)[0]
        txt_path = os.path.join(directory_path, file_id + '.txt')
        ann_path = os.path.join(directory_path, ann_file)

        if not os.path.exists(txt_path):
            continue

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            entities = {}
            relations = []
            
            with open(ann_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('T'):
                        parts = line.split('\t')
                        if len(parts) == 3:
                            entities[parts[0]] = parts[2]
                    elif line.startswith('R'):
                        parts = line.split('\t')
                        rel_type, arg1_id, arg2_id = None, None, None
                        
                        try:
                            relation_details = parts[1]
                            rel_type = relation_details.split(' ')[0]
                            arg1_match = arg1_re.search(relation_details)
                            arg2_match = arg2_re.search(relation_details)
                            if arg1_match and arg2_match:
                                arg1_id = arg1_match.group(1)
                                arg2_id = arg2_match.group(1)
                                relations.append((rel_type, arg1_id, arg2_id))
                        except Exception:
                            pass

            positive_pairs = set()
            
            for rel_type, arg1_id, arg2_id in relations:
                if arg1_id in entities and arg2_id in entities:
                    parsed_data.append({
                        'sentence': full_text,
                        'entity_1': entities[arg1_id],
                        'entity_2': entities[arg2_id],
                        'relation_type': rel_type
                    })
                    positive_pairs.add(tuple(sorted((arg1_id, arg2_id))))

            for id1, id2 in itertools.combinations(entities.keys(), 2):
                if tuple(sorted((id1, id2))) not in positive_pairs:
                    if id1 in entities and id2 in entities:
                        parsed_data.append({
                            'sentence': full_text,
                            'entity_1': entities[id1],
                            'entity_2': entities[id2],
                            'relation_type': 'NO_RELATION'
                        })
                        
        except Exception as e:
            print(f"Error parsing {ann_path}: {e}") 

    return pd.DataFrame(parsed_data)

# EXECUTION
if __name__ == "__main__":
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) 
    os.makedirs(OUTPUT_DIR)
    
    with open(LOG_FILE, 'w') as log_f:
        log_message("--- RegulaTome Data Preparation Report ---", log_f)
        log_message(f"Minimum Frequency Threshold: {MIN_FREQ_THRESHOLD}", log_f)
        log_message(f"Negative/Positive Ratio: {NEG_POS_RATIO}\n", log_f)
        
        # Process TRAINING Set 
        log_message("Parsing TRAINING data...", log_f)
        train_df = parse_brat_directory(os.path.join(CORPUS_PATH, 'train'))
        
        # Get counts of POSITIVE classes
        pos_counts = train_df[~train_df['relation_type'].isin(EXCLUDE_LIST)]['relation_type'].value_counts()
        
        # Determine which classes to keep (above threshold)
        classes_to_keep = pos_counts[pos_counts >= MIN_FREQ_THRESHOLD].index.tolist()
        
        log_message(f"Identified {len(classes_to_keep)} positive classes to keep (>= {MIN_FREQ_THRESHOLD} samples):", log_f)
        log_message(classes_to_keep, log_f)
        
        # Re-label rare/excluded classes as 'NO_RELATION'
        def clean_labels(rel_type, keep_list):
            if rel_type in keep_list:
                return rel_type
            else:
                return 'NO_RELATION'
                
        train_df['relation_type'] = train_df['relation_type'].apply(clean_labels, keep_list=classes_to_keep)
        
        # Balance TRAINING Set
        log_message("\nBalancing TRAINING data...", log_f)
        pos_df = train_df[train_df['relation_type'] != 'NO_RELATION']
        neg_df = train_df[train_df['relation_type'] == 'NO_RELATION']
        
        n_samples_to_keep = int(len(pos_df) * NEG_POS_RATIO)
        
        if len(neg_df) > n_samples_to_keep:
            neg_df_downsampled = neg_df.sample(n=n_samples_to_keep, random_state=RANDOM_STATE)
        else:
            neg_df_downsampled = neg_df 
            
        # Combine and shuffle
        train_balanced_df = pd.concat([pos_df, neg_df_downsampled]).sample(frac=1, random_state=RANDOM_STATE)
        
        train_path = os.path.join(OUTPUT_DIR, 'train_balanced.csv')
        train_balanced_df.to_csv(train_path, index=False)
        
        log_message("\n--- TRAINING SET ---", log_f)
        log_message(f"Saved balanced training set to {train_path}", log_f)
        log_message(f"Total samples: {len(train_balanced_df)} (Pos: {len(pos_df)}, Neg: {len(neg_df_downsampled)})", log_f)
        log_message("Relation Distribution:", log_f)
        log_message(train_balanced_df['relation_type'].value_counts().to_string(), log_f)
        
        # Process DEV Set
        log_message("\nParsing DEV data...", log_f)
        dev_df = parse_brat_directory(os.path.join(CORPUS_PATH, 'devel'))
        dev_df['relation_type'] = dev_df['relation_type'].apply(clean_labels, keep_list=classes_to_keep)
        
        dev_path = os.path.join(OUTPUT_DIR, 'dev_processed.csv')
        dev_df.to_csv(dev_path, index=False)
        
        log_message("\n--- DEV SET ---", log_f)
        log_message(f"Saved processed dev set to {dev_path}", log_f)
        log_message(f"Total samples: {len(dev_df)}", log_f)
        log_message("Relation Distribution:", log_f)
        log_message(dev_df['relation_type'].value_counts().to_string(), log_f)

        # Process TEST Set
        log_message("\nParsing TEST data...", log_f)
        test_df = parse_brat_directory(os.path.join(CORPUS_PATH, 'test'))
        test_df['relation_type'] = test_df['relation_type'].apply(clean_labels, keep_list=classes_to_keep)
        
        test_path = os.path.join(OUTPUT_DIR, 'test_processed.csv')
        test_df.to_csv(test_path, index=False)
        
        log_message("\n--- TEST SET ---", log_f)
        log_message(f"Saved processed test set to {test_path}", log_f)
        log_message(f"Total samples: {len(test_df)}", log_f)
        log_message("Relation Distribution:", log_f)
        log_message(test_df['relation_type'].value_counts().to_string(), log_f)
    
    print(f"\nData preparation complete! Report saved to {LOG_FILE}")