import csv
import tarfile
from pathlib import Path

def load_protein_synonyms(synonym_csv_path):
    full_name_map = {}  # protein name → UniProtID
    syn_map = {}        # synonym → UniProtID

    with open(synonym_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["UniProtID"].strip()
            pname = row["Protein"].strip()
            full_name_map[pname.lower()] = uid

            syns = row["Synonyms"].strip()
            if syns:
                for s in syns.split(";"):
                    syn = s.strip().lower()
                    if syn:
                        syn_map[syn] = uid

    return full_name_map, syn_map

def map_to_uniprot(text, full_name_map, syn_map):
    norm = text.lower().strip()
    if norm in full_name_map:
        return full_name_map[norm]
    if norm in syn_map:
        return syn_map[norm]
    # optional substring match
    for syn, uid in syn_map.items():
        if syn in norm or norm in syn:
            return uid
    return "NA"

# Parse ann content
def parse_ann_content(ann_text):
    entities = {}
    relations = []

    for line in ann_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Entities
        if line.startswith("T"):
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            tid, rest, text = parts
            fields = rest.split()
            etype = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            entities[tid] = {"text": text, "start": start, "end": end, "type": etype}

        # Relations
        elif line.startswith("R"):
            rid, rest = line.split("\t", 1)
            parts = rest.split()
            if len(parts) < 3:
                continue
            rtype = parts[0]
            arg1 = parts[1].split(":")[1]
            arg2 = parts[2].split(":")[1]
            relations.append({"id": rid, "type": rtype, "arg1": arg1, "arg2": arg2})

    return entities, relations

# Sentence extraction
def extract_sentence(text, start, end):
    left = text.rfind('.', 0, start) + 1
    right = text.find('.', end)
    if right == -1:
        right = len(text)
    return text[left:right].strip()

# Relation extraction logic
def extract_protein_relations_from_tar(tar_path, synonym_csv, output_file):
    full_name_map, syn_map = load_protein_synonyms(synonym_csv)
    results = []

    ann_files = {}
    txt_files = {}

    # Read all .ann and .txt files from the archive
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            pmid = Path(member.name).stem
            f = tar.extractfile(member)
            if not f:
                continue
            content = f.read().decode("utf-8")
            if member.name.endswith(".ann"):
                ann_files[pmid] = content
            elif member.name.endswith(".txt"):
                txt_files[pmid] = content

    # Process each PMID
    for pmid, ann_text in ann_files.items():
        if pmid not in txt_files:
            continue

        text = txt_files[pmid]
        entities, relations = parse_ann_content(ann_text)

        for r in relations:
            if r["arg1"] not in entities or r["arg2"] not in entities:
                continue
            p1 = entities[r["arg1"]]
            p2 = entities[r["arg2"]]

            # Keep only Protein–Protein relations
            if p1["type"] != "Protein" or p2["type"] != "Protein":
                continue

            # Map to UniProt IDs
            uid1 = map_to_uniprot(p1["text"], full_name_map, syn_map)
            uid2 = map_to_uniprot(p2["text"], full_name_map, syn_map)

            # Only keep if at least one protein is mentioned from synonym CSV
            if uid1 == "NA" and uid2 == "NA":
                continue

            sentence = extract_sentence(
                text,
                min(p1["start"], p2["start"]),
                max(p1["end"], p2["end"])
            )

            results.append({
                "Protein1": p1["text"],
                "Protein2": p2["text"],
                "Relation_type": r["type"],
                "UniProtId_1": uid1,
                "UniProtId_2": uid2,
                "PubMedID": pmid,
                "Sentence": sentence
            })

    # Save to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Protein1","Protein2","Relation_type",
            "UniProtId_1","UniProtId_2",
            "PubMedID","Sentence"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Extracted {len(results)} protein–protein relations → {output_file}")

if __name__ == "__main__":
    extract_protein_relations_from_tar(
        tar_path="RegulaTome-corpus.tar.gz",
        synonym_csv="protein_synonyms.csv",
        output_file="protein_relations.csv"
    )
