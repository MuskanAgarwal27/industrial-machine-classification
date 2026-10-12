import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load data
df = pd.read_csv("data/student_dataset.csv")
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

# Identical validation split for both models
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

models = {
    "Random Forest baseline": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting tuned": HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=15,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    ),
}

results = []

for name, classifier in models.items():
    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")

    model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", classifier),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")

    print(f"Accuracy: {accuracy:.5f}")
    print(f"Macro F1: {macro_f1:.5f}")
    print(classification_report(y_val, predictions, zero_division=0))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

results_df = pd.DataFrame(results).sort_values(
    "macro_f1", ascending=False
)

os.makedirs("outputs/model_comparison", exist_ok=True)
output_path = "outputs/model_comparison/final_model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nSaved to {output_path}")
