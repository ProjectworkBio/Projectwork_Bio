import pandas as pd
import requests
import urllib.parse

relations = pd.read_csv("final_ensemble_report_all_clean.csv")
syn = pd.read_csv("protein_synonyms.csv")

#Normalizing text columns
for c in syn.columns:
    syn[c] = syn[c].astype(str)
#building LOOKUP DICTIONARY
def get_all_names(row):
    names = []

    if row["ProteinName"]:
        names.append(row["ProteinName"])

    if row["UniProtID"]:
        names.append(row["UniProtID"])

    if row["GeneName"]:
        names.append(row["GeneName"])

    for col in ["ProteinSynonyms", "GeneSynonyms"]:
        if row[col] and row[col] != "nan":
            parts = [p.strip() for p in row[col].replace("|", ";").split(";")]
            names.extend(parts)

    names = [n for n in names if n != "" and n.lower() != "nan"]

    return list(dict.fromkeys(names))


lookup = {}
for _, row in syn.iterrows():
    key = row["ProteinName"].strip()
    lookup[key] = get_all_names(row)

#STRING API MATCHING
def find_string_id(name):
    #Returns STRING protein ID if name matches, else None
    species = "9606"  # means human
    url = f"https://string-db.org/api/json/get_string_ids?identifiers={urllib.parse.quote(name)}&species={species}"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()
    except:
        return None

    if isinstance(data, list) and len(data) > 0:
        return data[0]["stringId"]
    return None

#Trying all candidate names until STRING ID found

def try_all_names(original_name):
    #Try: ProteinName → GeneName → synonyms
    if original_name in lookup:
        for n in lookup[original_name]:
            found = find_string_id(n)
            if found:
                return found, n
    else:
        found = find_string_id(original_name)
        if found:
            return found, original_name

    return None, None
#Checking interaction between two STRING IDs

def check_string_interaction(id1, id2):
    url = f"https://string-db.org/api/json/network?identifiers={id1}%0d{id2}"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()
    except:
        return False, 0

    if isinstance(data, list) and len(data) > 0:

        score = max([edge.get("score", 0) for edge in data])
        return True, score
    return False, 0

#PROCESS ROWS AND GENERATE OUTPUT

results = []

for _, row in relations.iterrows():
    p1 = row["Entity_1_Name"].strip()
    p2 = row["Entity_2_Name"].strip()

#Geting STRING IDs
    id1, used1 = try_all_names(p1)
    id2, used2 = try_all_names(p2)

#Checking interaction

    if id1 and id2:
        found, score = check_string_interaction(id1, id2)
    else:
        found, score = False, 0

    results.append({
    "Entity_1_Name": p1,
    "Entity_2_Name": p2,
    "Ensemble_Result": row["Ensemble_Result"],
    "STRING_ID_1": id1,
    "STRING_ID_2": id2,
    "STRING_Found": found,
    "STRING_Score": score,
    "Used_Name_1": used1,
    "Used_Name_2": used2
    })

df_out = pd.DataFrame(results)
df_out.to_csv("STRING_validation_output.csv", index=False)
df_out.head()