import csv
from collections import Counter

csv_file = "protein_relations.csv"

counter = Counter()

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        counter[row["Relation_type"]] += 1

print("Relation types and their counts:")
for rtype, count in counter.most_common():
    print(f"{rtype}: {count}")
