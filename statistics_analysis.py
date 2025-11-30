import polars as pl
import re

syn_file = "protein_synonyms.csv"        
pubmed_files = "Result-v4\\all_results_cleaned.csv"      
output_file = "synonym_pubmed_frequencies.csv"
EXAMPLES_FILE = "synonym_examples.csv"

syn_df = pl.read_csv(syn_file)
pub_df = pl.read_csv(pubmed_files)
texts = pub_df["Relevant_Sentences"].to_list()

syn_exploded = (
    syn_df.with_columns(
        pl.col("Synonyms").str.split(";").alias("Syn_List")
    )
    .explode("Syn_List")
    .with_columns(pl.col("Syn_List").str.strip_chars().alias("Synonym"))
    .drop_nulls("Synonym")
    .select(["Protein", "Synonym"])
    .unique()
)

# build safe regex
def make_pattern(syn: str) -> re.Pattern:
    s = re.sub(r"\s+", " ", syn.strip())
    esc = re.escape(s)
    pattern = rf"(?<!\w){esc}(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)

# Count matches
rows = []
examples = []
MAX_EXAMPLES = 3

for rec in syn_exploded.iter_rows(named=True):
    protein = rec["Protein"]
    synonym = rec["Synonym"]
    if not synonym or synonym.strip() == "":
        continue
    pat = make_pattern(synonym)
    total = 0
    sample_sentences = []

    for t in texts:
        if not t:
            continue
        matches = pat.findall(str(t))
        if matches:
            total += len(matches)
            if len(sample_sentences) < MAX_EXAMPLES:
                sample_sentences.append(t)

    rows.append((protein, synonym, total))
    for s in sample_sentences:
        examples.append((protein, synonym, s))

# Convert to Polars DataFrames 
results = pl.DataFrame(rows, schema=["Protein", "Synonym", "Mentions"], orient="row")
results = results.sort("Mentions", descending=True)

examples_df = pl.DataFrame(examples, schema=["Protein", "Synonym", "Example_Sentence"], orient="row")

results.write_csv(output_file)
examples_df.write_csv(EXAMPLES_FILE)

print(f"Done. Saved counts to: {output_file}")
print(f"Example sentences saved to: {EXAMPLES_FILE}")
print("\nTop 10 synonyms by mention count:")
print(results.head(10))
