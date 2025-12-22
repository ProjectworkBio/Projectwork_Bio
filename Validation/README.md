# STRING-based Validation of Protein–Protein Relations

This repository contains a Python pipeline to validate predicted protein–protein relations using the STRING database.

## Overview

The pipeline evaluates whether protein pairs predicted by multiple relation extraction models
(BioBERT, PubMedBERT, BioLinkBERT, and the whole pipeline) are supported by known biological
interactions recorded in STRING.

The validation focuses on **interaction existence**, not on relation type semantics
(e.g., activation or inhibition).

---

## Input Files

### 1. final_report.csv
Contains predicted relations with columns:
- Entity_1
- Entity_2
- BioBERT_Relation
- PubMedBERT_Relation
- BioLinkBERT_Relation
- Ensemble_Result

### 2. protein_synonyms.csv
Provides alternative identifiers for proteins, including:
- ProteinName
- GeneName
- UniProtID
- ProteinSynonyms
- GeneSynonyms

---

## Methodology

1. Build a synonym lookup table for each protein.
2. Resolve STRING protein identifiers using:
   - Protein name
   - Gene name
   - Synonyms
3. Query STRING interaction network using STRING IDs.
4. Mark a protein pair as validated if **any interaction exists**.
5. Record STRING confidence scores and matched identifiers.

Caching is used extensively to improve runtime and reduce API calls.

---

## Output

### STRING_validation_output.csv

Contains:
- Model predictions
- STRING_Found (True / False)
- STRING confidence score
- Matched protein identifiers
- Synonyms used for matching

This output is used to compute:
- Accuracy
- Precision
- Recall
- F1-score
- Cross-model performance comparison

---

## Notes

- Species is restricted to human proteins (taxonomy ID: 9606).
- STRING validates interaction existence, not directionality or mechanism.
- High STRING coverage makes it suitable for immune-related protein interactions.

---

## Requirements

- Python ≥ 3.8
- pandas
- requests

---

## Reference

STRING database:
Szklarczyk et al., *Nucleic Acids Research*, 2021
