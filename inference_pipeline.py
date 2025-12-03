import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from itertools import combinations
import warnings
import os
from tqdm import tqdm

# CONFIGURATION
class Config:
    """Holds all static configuration for the pipeline."""
    # Base Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # File Paths
    INPUT_CSV = os.path.join(DATA_DIR, 'relevant_sentences.csv')
    PROTEIN_MAP_FILE = os.path.join(DATA_DIR, 'protein_dictionary.csv')
    OUTPUT_CSV = os.path.join(RESULTS_DIR, 'reports', 'ppi_report_BioLinkBERT_HC.csv')
    
    # Model Paths and Settings
    RE_MODEL_PATH = os.path.join(RESULTS_DIR,'models', 'BioLinkBERT', 'checkpoint-7587') 
    NER_STRATEGY = 'en_ner_jnlpba_md'   # en_ner_jnlpba_md or spacy_bionlp13cg
    CONFIDENCE_THRESHOLD = 0.80

    # Labels Mapping
    RELATION_MAPPING = {
        "Catalysis_of_phosphorylation": "Phosphorylates",
        "Catalysis_of_dephosphorylation": "Dephosphorylates",
        "Catalysis_of_ubiquitination": "Ubiquitinates",
        "Catalysis_of_SUMOylation": "Sumoylates",
        "Catalysis_of_acetylation": "Acetylates",
        "Catalysis_of_deacetylation": "Deacetylates",
        "Catalysis_of_methylation": "Methylates",
        "Catalysis_of_demethylation": "Demethylates",
        "Positive_regulation": "Activates",
        "Negative_regulation": "Inhibits",
        "Complex_formation": "Binds to / Interacts with",
        "Regulation_of_gene_expression": "Regulation_of_gene_expression",
        "Regulation_of_transcription": "Regulation_of_transcription",
        "Regulation": "Regulates",
        "NO_RELATION": "NO_RELATION"
    }
    
    # Labels to keep from the NER model
    NER_LABELS = {
        'en_ner_jnlpba_md': ["GENE_OR_GENE_PRODUCT", "SIMPLE_CHEMICAL", "CHEMICAL", "DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN"],
        'bert_d4data': ["Chemical", "Gene_or_gene_product", "Organism_substance", "Amino_acid", "Protein"]
    }

class NERExtractor:
    """Handles loading and running the NER model."""
    def __init__(self, strategy='en_ner_jnlpba_md'):
        self.strategy = strategy
        self.valid_labels = Config.NER_LABELS.get(strategy, [])
        print(f"Loading NER model: {strategy}")

        if strategy == 'en_ner_jnlpba_md':
            try:
                self.nlp = spacy.load("en_ner_bionlp13cg_md")
            except OSError:
                print("Error: Spacy 'en_ner_bionlp13cg_md' not found.")
                raise
        elif strategy == 'bert_d4data':
            try:
                self.pipeline = pipeline("ner", model="d4data/biomedical-ner-all", 
                                         aggregation_strategy="simple", device=0)
            except Exception as e:
                print(f"Error loading BERT NER pipeline: {e}")
                raise
        else:
            raise ValueError("Invalid NER strategy in Config.")

    def extract_entities(self, sentence):
        """Extracts entities from a single sentence."""            
        if self.strategy == 'en_ner_jnlpba_md':
            doc = self.nlp(sentence)
            return list(set([ent.text for ent in doc.ents if ent.label_ in self.valid_labels]))
        
        elif self.strategy == 'bert_d4data':
            ner_result = self.pipeline(sentence)
            return list(set([ent['word'] for ent in ner_result if ent['entity_group'] in self.valid_labels]))

class RelationExtractor:
    """Handles loading and running the fine-tuned RE model."""
    def __init__(self, model_path, device):
        print(f"Loading RE Model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.device = device
        self.sep_token = self.tokenizer.sep_token

    def predict(self, sentence, e1, e2):
        text_pair = f"{e1} {self.sep_token} {e2}"
        inputs = self.tokenizer(sentence, text_pair, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = torch.max(probs, dim=1)
        
        label_id = top_idx.item()
        score = top_prob.item()
        label = self.model.config.id2label[label_id]
        
        return label, score

class Normalizer:
    """Handles entity filtering and name normalization."""
    def __init__(self, map_file):
        self.protein_map, self.target_list = self._load_map_and_targets(map_file)

    def _load_map_and_targets(self, filepath):
        p_map = {}    # Maps alias -> (canonical_name, uniprot_id)
        t_list = set() 
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. No filtering or normalization.")
            return {}, []
            
        try:
            map_df = pd.read_csv(filepath)
            for _, row in map_df.iterrows():
                canonical_name = str(row['canonical_name']).strip()
                uniprot_id = str(row['uniprot_id']).strip()
                
                lookup_tuple = (canonical_name, uniprot_id)
                
                # 1. Add canonical name itself
                if canonical_name:
                    p_map[canonical_name.lower()] = lookup_tuple
                    t_list.add(canonical_name.lower())
                
                # 2. Add gene name
                if 'gene' in row and pd.notna(row['gene']):
                    gene_alias = str(row['gene']).lower().strip()
                    if gene_alias:
                        p_map[gene_alias] = lookup_tuple
                        t_list.add(gene_alias)

                # 3. Add all other synonyms
                if 'Synonyms' in row and pd.notna(row['Synonyms']):
                    synonyms_str = str(row['Synonyms']).strip()
                    if synonyms_str:
                        aliases = synonyms_str.split(';')
                        for alias in aliases:
                            alias = alias.strip().lower()
                            if alias:
                                p_map[alias] = lookup_tuple
                                t_list.add(alias)
                            
            print(f"Loaded {len(p_map)} total aliases for normalization.")
            print(f"Loaded {len(t_list)} unique aliases for filtering.")
            return p_map, t_list

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return {}, []

    def is_relevant(self, entity_text):
        if not self.target_list: return True
        t_lower = entity_text.lower()
        return any(target in t_lower or t_lower in target for target in self.target_list)

    def normalize(self, entity_text):
        if not self.protein_map: 
            return entity_text, "N/A"
        key = entity_text.lower()
        # 1. Exact match
        if key in self.protein_map:
            return self.protein_map[key]
        # 2. Substring match
        for alias, (canonical_name, uniprot_id) in self.protein_map.items():
            if alias in key:
                return canonical_name, uniprot_id 
        return entity_text, "N/A"

def main_pipeline():
    """Main function to run the complete pipeline."""
    warnings.filterwarnings("ignore")
    
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    normalizer = Normalizer(cfg.PROTEIN_MAP_FILE)
    ner_model = NERExtractor(strategy=cfg.NER_STRATEGY)
    re_model = RelationExtractor(cfg.RE_MODEL_PATH, device)
    
    df = pd.read_csv(cfg.INPUT_CSV)
    print(f"Processing {len(df)} sentences from {cfg.INPUT_CSV}...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Relations"):
        sentence = str(row.get('Relevant_Sentence', ''))
        pmid = row.get('PubMedID', 'Unknown')

        entities_raw = ner_model.extract_entities(sentence)
        # if len(entities_raw) < 2: continue

        for e1_raw, e2_raw in combinations(entities_raw, 2):
            
            if not (normalizer.is_relevant(e1_raw) or normalizer.is_relevant(e2_raw)):
                continue
        
            raw_label, conf = re_model.predict(sentence, e1_raw, e2_raw)
            
            # Threshold and Mapping
            if conf >= cfg.CONFIDENCE_THRESHOLD:
                #mapped_label = cfg.RELATION_MAPPING.get(raw_label, raw_label)
                mapped_label = raw_label
                # Normalize
                e1_norm, e1_id = normalizer.normalize(e1_raw)
                e2_norm, e2_id = normalizer.normalize(e2_raw)
                    
                results.append({
                        "PubMedID": pmid,
                        "Entity_1_Normalized": e1_norm,
                        "Entity_1_UniProt": e1_id,
                        "Relation": mapped_label,
                        "Entity_2_Normalized": e2_norm,
                        "Entity_2_UniProt": e2_id,
                        "Confidence": round(conf, 4),
                        "Entity_1": e1_raw,
                        "Entity_2": e2_raw,
                        "Sentence": sentence
                })
                '''
                # If we only need real relations
                if mapped_label != "NO_RELATION":
                    # Normalize
                    e1_norm, e1_id = normalizer.normalize(e1_raw)
                    e2_norm, e2_id = normalizer.normalize(e2_raw)
                    
                    results.append({
                        "PubMedID": pmid,
                        "Entity_1_Normalized": e1_norm,
                        "Entity_1_UniProt": e1_id,
                        "Relation": mapped_label,
                        "Entity_2_Normalized": e2_norm,
                        "Entity_2_UniProt": e2_id,
                        "Confidence": round(conf, 4),
                        "Entity_1": e1_raw,
                        "Entity_2": e2_raw,
                        "Sentence": sentence
                    })
                '''

    print("\nProcessing Complete.")
    if results:
        out_df = pd.DataFrame(results).sort_values("Confidence", ascending=False)
        out_df.to_csv(cfg.OUTPUT_CSV, index=False)
        print(f"{len(out_df)} normalized relations saved to {cfg.OUTPUT_CSV}")
    else:
        print("No relations found.")

if __name__ == "__main__":
    main_pipeline()