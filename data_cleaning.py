import polars as pl

input_path = "Result-v5/all_results_concatenated.csv"
output_path = "Result-v5/all_results_cleaned.csv"

df = pl.read_csv(input_path)

print(f"Loaded dataset with {df.height} rows and {df.width} columns")
original_length = df.height

# Trim whitespace from all columns
df = df.with_columns([
    pl.col(col).str.strip_chars() if df[col].dtype == pl.Utf8 else pl.col(col)
    for col in df.columns
])

# Replace empty strings or "NA" / "None" with nulls
invalid_values = ["", "na", "n/a", "none", "null"]

# Count rows that contain *any* of those invalid values before cleaning
affected_rows = df.filter(
    pl.any_horizontal(
        [
            pl.col(col).str.to_lowercase().is_in(invalid_values)
            for col in df.columns
            if df[col].dtype == pl.Utf8
        ]
    )
).height

print(f"Rows affected by invalid string cleanup: {affected_rows}")
df = df.with_columns([
    pl.when(pl.col(col).str.to_lowercase().is_in(invalid_values))
    .then(None)
    .otherwise(pl.col(col))
    .alias(col)
    for col in df.columns if df[col].dtype == pl.Utf8
])

# Drop rows where essential columns are missing
essential_cols = ["PubMedID", "Relevant_Sentence"]
df = df.drop_nulls(subset=essential_cols)

# Normalize spacing and punctuation in "Relevant_Sentences"
df = df.with_columns(
    pl.col("Relevant_Sentence")
    .str.replace_all(r"\s*\|\|\s*", " || ")  # ensure consistent separator
    .str.replace_all(r"\s{2,}", " ")         # collapse extra spaces
    .str.strip_chars()                       # trim edges
    .alias("Relevant_Sentence")
)

# Remove all variants of "(ABSTRACT TRUNCATED...)"
affected_rows = df.filter(
    pl.col("Relevant_Sentence").str.contains(r"\(ABSTRACT TRUNCATED(?: AT \d+ WORDS)?\)")
).height
print(f"Rows affected by cleanup (truncated abstracts): {affected_rows}")

substring = "(ABSTRACT TRUNCATED AT "
df = df.with_columns(
    pl.col("Relevant_Sentence")
    .str.replace_all(r"\(ABSTRACT TRUNCATED(?: AT \d+ WORDS)?\)", "")
    # Clean up leftover extra spaces
    .str.replace_all(r"\s{2,}", " ")
    .str.strip_chars()
    .alias("Relevant_Sentence")
)

# Remove duplicate rows (exact duplicates)
df = df.unique(subset=["PubMedID", "Matched_Proteins", "Relevant_Sentence"])

# Save cleaned dataset
df.write_csv(output_path)
print(f"Cleaned dataset saved to '{output_path}' with {df.height} rows")
print(f"Cleaned dataset size: {df.height}")
print("\nExample cleaned rows:")
print(df.head(5))

