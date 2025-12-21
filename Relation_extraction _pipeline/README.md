# Automated Biomedical Relation Extraction

## Overview

This code implements a hybrid architecture for extracting fine-grained protein-protein interactions from biomedical literature.

The pipeline consists of:

1. **Data Processing**: Merges RegulaTome (Molecular) and BioRED (Clinical) into a Unified Super-Corpus.

2. **The Realtion Extractor**: An ensemble of BioLinkBERT, PubMedBERT, and BioBERT encoders fine-tuned to classify relations with high recall.

3. **The Judge**: A BioMistral-7B generative LLM that validates high-confidence predictions to remove false positives.

## Data Sources & Acknowledgements

This software utilizes the following datasets. If you use this code, please cite the original authors:

1. **RegulaTome**
> Nastou, K. C., et al. (2024). RegulaTome: a corpus of typed, directed, and signed relations between biomedical entities. https://doi.org/10.5281/zenodo.10808330

3. **BioRED**

>Luo L., Lai P. T., Wei C. H., Arighi C. N. and Lu Z. (2022). BioRED: A Rich Biomedical Relation Extraction Dataset. Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbac282

## Usage

Use the master script main.py to run any stage of the pipeline.

1. Download, Parse & Balance Data:
```bash
python main.py parser
```


2. Preprocess (Add Entity Markers):
```bash
python main.py preprocess --file data/processed/train.csv

```



3. Train the Extractor Ensemble:
```bash
python main.py train --epochs 6 --lr 2e-5

```



4. Run Evaluation on Test File:
```bash
python main.py evaluate

```



5. Run Inference :
```bash
python main.py predict --input "results_filtered.csv"

```



6. Apply Ensemble Logic & BioMistral Judge:
```bash
python main.py ensemble --input "raw_predictions.csv"
python main.py validate_hard_cases

```



7. Generate Final Report:
```bash
python main.py results_report
```
