import os
import pandas as pd

# --- CONFIGURATION ---
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
ENSEMBLE_FILE = os.path.join(
    BASE_DIR, "results", "analysis", "final_ensemble_report.csv"
)

JUDGE_FILE = os.path.join(
    BASE_DIR, "results", "analysis", "final_adjudicated_relations.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "results", "analysis", "final_report.csv"
)

CORE_COLS = [
    "PubMedID",
    "Entity_1",
    "Entity_2",
    "Sentence"
]

MODEL_COLS = [
    "BioBERT_Relation",
    "PubMedBERT_Relation",
    "BioLinkBERT_Relation"
]

def merge_result():
    if not os.path.exists(ENSEMBLE_FILE):
        raise FileNotFoundError(f"Missing ensemble file: {ENSEMBLE_FILE}")

    if not os.path.exists(JUDGE_FILE):
        raise FileNotFoundError(f"Missing judge file: {JUDGE_FILE}")

    print("Loading ensemble predictions...")
    df_ens = pd.read_csv(ENSEMBLE_FILE)

    print("Loading judge adjudications...")
    df_judge = pd.read_csv(JUDGE_FILE)


    if "Ensemble_Relation" in df_ens.columns:
        df_ens = df_ens.rename(columns={"Ensemble_Relation": "Relation"})

    else:
        raise ValueError("Ensemble relation column not found.")

    ens_cols = CORE_COLS + MODEL_COLS + ["Relation"]
    df_ens = df_ens[ens_cols].copy()
    df_ens["Source"] = "ENCODER_ENSEMBLE"


    if "Final_Adjudicated_Relation" not in df_judge.columns:
        raise ValueError("Final_Adjudicated_Relation missing in judge file.")

    judge_cols = CORE_COLS + MODEL_COLS + ["Final_Adjudicated_Relation"]
    df_judge = df_judge[judge_cols].copy()
    df_judge = df_judge.rename(
        columns={"Final_Adjudicated_Relation": "Relation"}
    )
    df_judge["Source"] = "LLM_JUDGE"


    final_df = pd.concat(
        [df_ens, df_judge],
        axis=0,
        ignore_index=True
    )


    final_df = final_df.sort_values(
        by=["PubMedID", "Entity_1", "Entity_2"],
        kind="stable"
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal unified report created.")
    print(f"Total relations: {len(final_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(final_df["Source"].value_counts())

    