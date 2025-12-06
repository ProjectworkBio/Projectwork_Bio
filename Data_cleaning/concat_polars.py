import os
import polars

result_folder = "Result-v4"
output_file = os.path.join(result_folder, "all_results_concatenated.csv")

# Collect all CSV files
csv_files = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith("_2prot_sentences.csv")]

lf = polars.scan_csv(csv_files)
lf.sink_csv(output_file)
