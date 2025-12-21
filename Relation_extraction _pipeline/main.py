import argparse
import subprocess
import os
import sys

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
SCRIPTS = {
    "parser": os.path.join(BASE_DIR, 'data_parser.py'),  
    "preprocess": os.path.join(BASE_DIR, 'preprocess.py'),  
    "train": os.path.join(BASE_DIR, 'train_models.py'),   
    "evaluate": os.path.join(BASE_DIR, 'evaluate_models.py'),        
    "predict": os.path.join(BASE_DIR, 'run_inference.py'),  
    "ensemble": os.path.join(BASE_DIR, 'run_ensemble_logic.py'), 
    "validate_hard_cases": os.path.join(BASE_DIR, 'run_biomistral_judge.py'),   
    "results_report": os.path.join(BASE_DIR, 'csvToJson.py'),   
}

def run_script(script_name, args=[]):
    """Executes a python script as a subprocess."""
    if not os.path.exists(script_name):
        print(f"Error: Could not find script '{script_name}'")
        print(f"Please make sure it is in the same folder as main.py")
        sys.exit(1)

    print(f"\ Starting module: {script_name}...")
    print(f"Arguments: {args}")
    print("-" * 50)
    
    # Construct command: python script.py [args]
    cmd = [sys.executable, script_name] + args
    
    try:
        # Run and wait for completion
        subprocess.check_call(cmd)
        print("-" * 50)
        print(f"Module '{script_name}' finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Module '{script_name}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Biomedical Relation Extraction System")
    subparsers = parser.add_subparsers(dest="mode", help="Select the pipeline stage", required=True)
    
    # Parser Mode
    parser_pars = subparsers.add_parser("parser", help="Parse RegulaTome/BioRED and create training data")

    # Preprocess Mode
    parser_prep = subparsers.add_parser("preprocess", help="Preprocess test or raw data for relation extraction")
    parser_prep.add_argument("--file", type=str, default="data/results_filtered.csv", help="Data file path")


    # Train Mode
    parser_train = subparsers.add_parser("train", help="Train the Bert models on processed data")
    parser_train.add_argument("--epochs", type=str, default="6", help="Number of training epochs")
    parser_train.add_argument("--lr", type=str, default="4e-6", help="Model Learning Rate")
    
    # Evaluate/Benchmark Mode
    parser_eval = subparsers.add_parser("evaluate", help="Run model comparison and ensemble evaluation")
    
    # Predict Mode
    parser_pred = subparsers.add_parser("predict", help="Extract relations from a new csv file")
    parser_pred.add_argument("--input", type=str, default="data/processed_data.csv", help="Path to input file to analyze")
    
    # Run Ensemble 
    parser_ens = subparsers.add_parser("ensemble", help="Run ensemble logic")

    # Run Biomistral For Hard Cases
    parser_ens = subparsers.add_parser("validate_hard_cases", help="Run validation by biomistral")

    # Run Biomistral For Hard Cases
    parser_res = subparsers.add_parser("results_report", help="Get final results from all models and the whole pipeline")


    args = parser.parse_args()

    # --- DISPATCHER ---
    if args.mode == "parser":
        run_script(SCRIPTS["parser"])

    elif args.mode == "preprocess":
        run_script(SCRIPTS["preprocess"],  ["--file", args.file])
        
    elif args.mode == "train":
        run_script(SCRIPTS["train"],  ["--epochs", args.epochs, "--lr", args.lr])
        
    elif args.mode == "evaluate":
        run_script(SCRIPTS["evaluate"])
        
    elif args.mode == "predict":
        run_script(SCRIPTS["predict"], ["--input", args.input])

    elif args.mode == "ensemble":
        run_script(SCRIPTS["ensemble"])
    
    elif args.mode == "validate_hard_cases":
        run_script(SCRIPTS["validate_hard_cases"])

    elif args.mode == "results_report":
        run_script(SCRIPTS["results_report"])

if __name__ == "__main__":
    main()

"""
DATASET ACKNOWLEDGEMENT:
This software utilizes the following datasets for training:
1. RegulaTome: Nastou et al. (2024) - https://doi.org/10.5281/zenodo.10808330
2. BioRED: Luo et al. (2022) - https://doi.org/10.1093/bib/bbac282
"""