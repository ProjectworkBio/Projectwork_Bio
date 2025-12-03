import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "final_ppi_report.csv"

df = pd.read_csv(INPUT_CSV)

plt.figure(figsize=(10, 6))
sns.histplot(df['Confidence'], bins=20, kde=True, color='blue')

plt.title('Model Confidence Distribution')
plt.xlabel('Confidence Score (0.0 - 1.0)')
plt.ylabel('Count of Predictions')
plt.axvline(x=0.80, color='r', linestyle='--', label='Your Threshold (0.80)')
plt.legend()

print("Plotting histogram...")
plt.savefig("validation_confidence_histogram.png")
plt.show()

print(f"Mean Confidence: {df['Confidence'].mean():.4f}")
print("Looking for a 'U-shaped' curve (peaks at high confidence). A bell curve in the middle indicates confusion.")