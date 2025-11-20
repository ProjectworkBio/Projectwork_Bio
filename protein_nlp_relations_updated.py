
import pandas as pd
import tarfile
import re

# -----------------------------------------
# 1. Load protein list
# -----------------------------------------
proteins = [line.strip() for line in open('proteins_original.txt') if line.strip()]

# -----------------------------------------
# 2. Load synonyms file
# -----------------------------------------
syn = pd.read_csv('protein_synonyms.csv', sep="\t")

# Build synonym map: each protein → set of synonyms
synmap = {}
for _, row in syn.iterrows():
    syns = []
    if isinstance(row["Synonyms"], str):
        syns = [s.strip() for s in row["Synonyms"].split(";")]
    synmap[row["Protein"]] = set([row["Protein"]] + syns)

# Create reverse lookup: term → protein
term2prot = {}
for prot, syns in synmap.items():
    for t in syns:
        term2prot[t.lower()] = prot

# -----------------------------------------
# Helper: simple sentence splitter
# -----------------------------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

# -----------------------------------------
# 3. Define simple verb patterns
# -----------------------------------------
patterns = {
    "activates": r"\bactivates?\b|\binduces?\b|\bstimulates?\b",
    "inhibits": r"\binhibits?\b|\bsuppresses?\b|\bblocks?\b",
    "binds": r"\bbinds?\b|\binteracts?\b",
    "regulates": r"\bregulates?\b|\bmodulates?\b",
}

# -----------------------------------------
# 4. Read the corpus and extract relations
# -----------------------------------------
records = []

with tarfile.open('RegulaTome-corpus.tar.gz') as tar:
    for name in tar.getnames():
        if name.endswith('.txt'):

            # Extract PubMed ID from filename
            base = name.split("/")[-1]
            m = re.match(r"(\d+)", base)
            if not m:
                continue
            pmid = m.group(1)

            # Read text
            txt = tar.extractfile(name).read().decode(errors='ignore')
            sentences = split_sentences(txt)

            # Process each sentence
            for sent in sentences:
                low = sent.lower()

                # Which proteins appear in the sentence?
                present = set()
                for term, prot in term2prot.items():
                    if term and term in low:
                        present.add(prot)
                present = sorted(present)

                # Must have at least 2 proteins
                if len(present) < 2:
                    continue

                # For each protein pair, determine relation type
                for i in range(len(present)):
                    for j in range(i+1, len(present)):
                        p1, p2 = present[i], present[j]

                        # Default: unknown
                        relation = "unknown"
                        for rel, pat in patterns.items():
                            if re.search(pat, low):
                                relation = rel
                                break

                        # Store result
                        records.append([
                            p1,           # Protein 1
                            p2,           # Protein 2
                            relation,     # relation type
                            p1,           # ACTUAL protein 1
                            p2,           # ACTUAL protein 2
                            pmid,         # PubMed ID
                            sent.strip()  # full sentence
                        ])

# -----------------------------------------
# 5. Create dataframe & save output
# -----------------------------------------
df = pd.DataFrame(records, columns=[
    "Protein 1", "Protein 2", "relation", "ACTUAL protein 1",
    "Actual protein 2", "PubMedID", "Sentence"
])

df.to_csv('protein_nlp_relations.csv', index=False)

print("Saved as protein_nlp_relations.csv with", len(df), "relations.")
