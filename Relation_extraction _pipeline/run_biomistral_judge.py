import os
import warnings
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)

from tqdm import tqdm

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
INPUT_CSV = os.path.join(
    BASE_DIR, "results", "analysis", "hard_cases_for_llm_judge.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "results", "analysis", "final_adjudicated_relations.csv"
)

MODEL_ID = "BioMistral/BioMistral-7B"
SEED = 42

warnings.filterwarnings("ignore")
set_seed(SEED)

assert torch.cuda.is_available(), (
    "CUDA GPU not detected. BioMistral-7B requires a GPU."
)

# LOAD BioMistral (4-bit)
print(f"Loading {MODEL_ID} (4-bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print("BioMistral Judge is ready.")

def adjudicate_once(sentence, e1, e2, options):
    options_str = ", ".join([f"'{o}'" for o in options])

    prompt = f"""
<s>[INST]
You are a strict biomedical relation classifier.

Given a sentence containing two protein entities and a list of candidate relation types,
select EXACTLY one relation from the candidate list.

Return ONLY the label.

Sentence: "{sentence}"
Entity_1: "{e1}"
Entity_2: "{e2}"
Candidates: [{options_str}]
[/INST]
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("[/INST]")[-1].strip()

    # Strict normalization
    for opt in options:
        if answer.lower() == opt.lower():
            return opt

    return "NO_RELATION"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Adjudicating {len(df)} hard cases...")

relation_cols = [c for c in df.columns if c.endswith("_Relation")]

final_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    sentence = row["Sentence"]
    e1 = row["Entity_1"]
    e2 = row["Entity_2"]

    options = sorted(
        set(row[c] for c in relation_cols if pd.notna(row[c]))
    )

    if "NO_RELATION" not in options:
        options.append("NO_RELATION")

    final_label = adjudicate_once(sentence, e1, e2, options)

    out = row.to_dict()
    out.update({
        "Final_Adjudicated_Relation": final_label,
    })

    final_rows.append(out)


out_df = pd.DataFrame(final_rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
out_df.to_csv(OUTPUT_CSV, index=False)

print("\nAdjudication complete.")
print(out_df["Final_Adjudicated_Relation"].value_counts())
print(f"Saved → {OUTPUT_CSV}")
