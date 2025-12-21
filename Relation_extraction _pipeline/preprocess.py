import pandas as pd
import argparse
import os
import re

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR)


def add_entity_markers(sentence, e1, e2):
    """
    Adds [E1][/E1] and [E2][/E2] markers to a sentence.
    Returns None if entities cannot be safely marked.
    """
    # Escape entities for regex
    e1_esc = re.escape(e1)
    e2_esc = re.escape(e2)

    # Find matches
    e1_match = re.search(rf"\b{e1_esc}\b", sentence)
    e2_match = re.search(rf"\b{e2_esc}\b", sentence)

    # If either entity not found → reject
    if not e1_match or not e2_match:
        return None

    # Prevent overlapping entities
    ''' 
    example : 
    entity1 : Tumor necrosis factor
    entity2 : Tumor necrosis factor alpha
    Tumor necrosis factor alpha (TNF alpha) is suggested to be of importance in the pathogenesis of inflammatory diseases.
    '''
    if e1_match.start() <= e2_match.end() and e2_match.start() <= e1_match.end():
        return None

    # Insert markers from right to left (important!)
    spans = sorted(
        [(e1_match.start(), e1_match.end(), "E1"),
         (e2_match.start(), e2_match.end(), "E2")],
        reverse=True
    )

    marked_sentence = sentence
    for start, end, tag in spans:
        marked_sentence = (
            marked_sentence[:start]
            + f"[{tag}] "
            + marked_sentence[start:end]
            + f" [/{tag}]"
            + marked_sentence[end:]
        )
    return marked_sentence



def main():
    parser = argparse.ArgumentParser(description="Process input CSV file.")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the input CSV file to be processed"
    )
    args = parser.parse_args()
    if not os.path.exists(args.file):
            print(f"Error: The file '{args.file}' does not exist.")
            return

    print(f"Loading data from: {args.file}")
    df = pd.read_csv(args.file)
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    marked_sentences = []
    dropped = 0

    # Force entity columns to string and drop NaNs
    df = df.dropna(subset=["Sentence", "Matched_Protein_Name", "Other_protein"])

    df["Matched_Protein_Name"] = df["Matched_Protein_Name"].astype(str)
    df["Other_protein"] = df["Other_protein"].astype(str)
    df["Sentence"] = df["Sentence"].astype(str)


    for _, row in df.iterrows():
        sent = row["Sentence"]
        e1 = row["Matched_Protein_Name"]
        e2 = row["Other_protein"]

        marked = add_entity_markers(sent, e1, e2)

        if marked is None:
            dropped += 1
            marked_sentences.append(None)
        else:
            marked_sentences.append(marked)

    df["marked_text"] = marked_sentences
    # Drop failed rows
    df = df.dropna(subset=["marked_text"])
    out_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Dropped {dropped} rows due to missing or ambiguous entities")
    print(f"Processed file is saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()