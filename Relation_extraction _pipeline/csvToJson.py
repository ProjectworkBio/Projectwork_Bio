import os
import json
import pandas as pd
from merge_results import merge_result

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
merge_result()
INPUT_CSV = os.path.join(BASE_DIR, 'results', 'analysis', 'final_report.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'results', 'analysis', 'ppi_final.json')

# Load CSV
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

# Validate required columns
required = [
    "PubMedID",
    "Sentence",         # -> title
    "Entity_1",         # -> relations[].source
    "Entity_2",         # -> relations[].target
    "Relation"     # -> relations[].type
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}\n"
                     f"Found columns: {list(df.columns)}")

# Clean up strings and fill NaNs
for col in required:
    if df[col].dtype == object:
        df[col] = df[col].fillna("").astype(str).str.strip()
    else:
        df[col] = df[col].fillna("")

# Build the grouped JSON structure
records = []
for pmid, g in df.groupby("PubMedID", sort=False):
    title_value = next((t for t in g["Sentence"].tolist() if t), "").strip()
    if not title_value:
        title_value = f"PMID {pmid}"

    rels = []
    seen = set()
    for _, row in g.iterrows():
        
        tup = (
            row["Entity_1"],
            row["Entity_2"],
            row["Relation"]
        )
        if tup in seen:
            continue
        seen.add(tup)
        rels.append({
            "source": tup[0],
            "target": tup[1],
            "type":   tup[2]
        })

    records.append({
        "pmid": str(pmid),
        "title": title_value,
        "relations": rels
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(records)} records to: {OUTPUT_JSON}")
