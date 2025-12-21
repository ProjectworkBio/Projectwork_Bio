# Data Extraction and Summarization using LLMs
This project, developed in collaboration with Biogenity, focuses on designing and implementing a secure and scalable data extraction and summarization system using Large Language Models (LLMs). The solution integrates data collection, preprocessing and summarization pipelines, to ensure accuracy and efficiency. Various LLM architectures are explored and evaluated in their performance on domain-specific datasets, aiming to enhance the usability and reliability of automated knowledge extraction tools.

# Installation of dependencies
1. Install UV from  [astral.sh](https://docs.astral.sh/uv/getting-started/installation/)
2. Once installed, open the project folder in the terminal and run `uv sync`
3. Then you can run the .py file with `uv run` or in you IDE select the .venv folder to use the correct environment with all the dependencies

# Data_extraction_pipeline
1. **extract_baseline.py**: The pipeline starts with downloading the complete PubMed Baseline dataset of the current year. The goal of the code is to retrieve all compressed XML files from the NCBI baseline repository, store them locally, and prepare them for later transformation and sentence-level extraction.
2. **transform_xml_to_csv.py**: During the tranformation phase the PubMed IDs and their associated abstract texts are extracted from every compressed XML file and saved to CSV files.
3. **filter.py**: Relevant sentences are filtered based on the presence of at least one protein from a predefined set of 45 proteins of interest.
4. **updates.py**: The daily update files are downloaded and compared with the processed baseline dataset. For consistency, the up-date dataset undergoes the same preprocessing steps applied to the baseline: data extraction and filtering. After preprocessing, the resulting CSV files are merged using the PubMedID as a unique identifier.
5. **statistics.py**: The filtered dataset undergoes statistical analysis.

# 

