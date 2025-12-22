# Data Extraction and Summarization using LLMs
This project, developed in collaboration with Biogenity, focuses on designing and implementing a secure and scalable data extraction and summarization system using Large Language Models (LLMs). The solution integrates data collection, preprocessing and summarization pipelines, to ensure accuracy and efficiency. Various LLM architectures are explored and evaluated in their performance on domain-specific datasets, aiming to enhance the usability and reliability of automated knowledge extraction tools.

# Installation of dependencies
1. Install UV from  [astral.sh](https://docs.astral.sh/uv/getting-started/installation/)
2. Once installed, open the project folder in the terminal and run `uv sync`
3. Then you can run the .py file with `uv run` or in you IDE select the .venv folder to use the correct environment with all the dependencies

# Data Extraction Pipeline
**Directory:** [`/Data_extraction_pipeline`](./Data_extraction_pipeline)
1. **extract_baseline.py**: The pipeline starts with downloading the complete PubMed Baseline dataset of the current year. The goal of the code is to retrieve all compressed XML files from the NCBI baseline repository, store them locally, and prepare them for later transformation and sentence-level extraction.
2. **transform_xml_to_csv.py**: During the tranformation phase the PubMed IDs and their associated abstract texts are extracted from every compressed XML file and saved to CSV files.
3. **filter.py**: Relevant sentences are filtered based on the presence of at least one protein from a predefined set of 45 proteins of interest.
4. **updates.py**: The daily update files are downloaded and compared with the processed baseline dataset. For consistency, the up-date dataset undergoes the same preprocessing steps applied to the baseline: data extraction and filtering. After preprocessing, the resulting CSV files are merged using the PubMedID as a unique identifier.
5. **statistics.py**: The filtered dataset undergoes statistical analysis.


# Relation Extraction Pipeline
**Directory:** [`/Relation_Extraction_Pipeline`](./Relation_Extraction_Pipeline)

This module contains the Deep Learning logic for extracting specific relations from biomedical papers. It accepts input from the data extraction pipeline and extarct precise mechanisms (e.g., Phosphorylation, Up-regulation) from the input file.

# Validation (STRING-based)
**Directory:** [`/Validation`](./Validation)

This module ensures the biological plausibility of the extracted relations by cross-referencing them with external databases.

# OmicsViz (Visualization Dashboard)
**Directory:** [`/OmicsViz`](./omicsviz)

A web-based interactive dashboard designed to explore the extracted protein/disease relations.

## Workflow

To replicate the full project pipeline, execute the modules in the following order:

1.  **Dat Extraction:** Navigate to `Data_extraction_pipeline` and run the extraction scripts to generate input data for relation extarction.
2. **Extract Relations:** Navigate to `Relation_Extraction_Pipeline` and run the extraction scripts to generate raw prediction data from your text corpus.
3.  **Validate Results:** Use the scripts in `Validation` to score the predictions against the STRING database and filter out biologically unlikely pairs.
4.  **Visualize:** Load the final validated results into the `OmicsViz` application to explore the knowledge graph interactively.

> **Note**: Please refer to the specific README.md file inside each sub-directory for detailed installation requirements and execution instructions for that specific module.

