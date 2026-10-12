import pandas as pd
from pathlib import Path

results = [
    ["HistGradientBoosting", "Baseline", 0.8865625, 0.875667],
    ["HistGradientBoosting", "Tuned", 0.888125, 0.877019],
    ["Random Forest", "Baseline", 0.886875, 0.875817],
    ["Random Forest", "Tuned for accuracy", 0.884062, 0.872303],
    ["Random Forest", "Tuned for Macro F1", 0.882812, 0.871840],
    ["Extra Trees", "Baseline", 0.878125, 0.864290],
    ["Logistic Regression", "Baseline", 0.822500, 0.804281],
    ["Gaussian Naive Bayes", "Baseline", 0.765000, 0.735121],
]

df = pd.DataFrame(
    results,
    columns=["Model", "Version", "Accuracy", "Macro_F1"]
)

df = df.sort_values("Macro_F1", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", range(1, len(df) + 1))

print("\n========== FINAL MODEL RESULTS ==========\n")
print(df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.5f}".format,
        "Macro_F1": "{:.5f}".format,
    }
))

output = Path("outputs/model_comparison")
output.mkdir(parents=True, exist_ok=True)

file_path = output / "final_all_models_summary.csv"
df.to_csv(file_path, index=False)

print(f"\nSaved results to: {file_path}")
