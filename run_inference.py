import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import warnings

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'data_v2.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'results', 'reports', 'ppi_report_v3_PubMedBERT.csv')
MODEL_PATH = "results/models/PubMedBERT/checkpoint-7587"
CONFIDENCE_THRESHOLD = 0.80

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
    "Regulation_of_gene_expression": "Regulates",
    "Regulation_of_transcription": "Regulates",
    "Regulation": "Regulates",
    "NO_RELATION": "NO_RELATION"
}

def main():
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    
    if not os.path.exists(INPUT_CSV):
        print("Please run prepare_inference_data.py first!")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Running inference on {len(df)} pairs...")
    
    final_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sentence = str(row['Sentence'])
        e1 = str(row['Entity_1_Raw'])
        e2 = str(row['Entity_2_Raw'])
        
        text_pair = f"{e1} {tokenizer.sep_token} {e2}"
        inputs = tokenizer(sentence, text_pair, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = torch.max(probs, dim=1)
        
        raw_label = model.config.id2label[top_idx.item()]
        conf = top_prob.item()

        mapped_label = RELATION_MAPPING.get(raw_label, raw_label)
        result_row = row.to_dict()
        result_row['Predicted_Relation'] = raw_label
        result_row['Confidence'] = round(conf, 4)
        result_row['Original_Label'] = raw_label
        final_results.append(result_row)
        
        # Filter and Mapping
        '''
        if conf >= CONFIDENCE_THRESHOLD:
            mapped_label = RELATION_MAPPING.get(raw_label, raw_label)
            
            if mapped_label != "NO_RELATION":
                # Save the result (copying metadata from input row)
                result_row = row.to_dict()
                result_row['Predicted_Relation'] = mapped_label
                result_row['Confidence'] = round(conf, 4)
                result_row['Original_Label'] = raw_label
                final_results.append(result_row)
        '''
                
    if final_results:
        out_df = pd.DataFrame(final_results)
        cols = ['PubMedID', 'Entity_1_Name', 'Predicted_Relation', 'Entity_2_Name', 'Confidence', 'Entity_1_ID', 'Entity_2_ID', 'Sentence']
        out_df = out_df[cols + [c for c in out_df.columns if c not in cols]]
        
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Success! {len(out_df)} relations saved to {OUTPUT_CSV}")
    else:
        print("No relations found.")

if __name__ == "__main__":
    main()
