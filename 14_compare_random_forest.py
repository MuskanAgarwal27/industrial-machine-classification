import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Load labelled data
train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]

# Use exactly the same split for both models
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# Baseline parameters from the original model
models = {
    "Baseline Random Forest": {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "Tuned Random Forest": {
        "n_estimators": 500,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "max_depth": 20,
        "class_weight": None,
    },
}

results = []

for name, params in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params,
        )),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(y_valid, predictions, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
    })

    print(f"\n{'=' * 55}")
    print(name)
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Macro F1: {macro_f1:.5f}")
    print("\nPer-class report:")
    print(classification_report(y_valid, predictions, zero_division=0))

results_df = pd.DataFrame(results)
print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))

import os
os.makedirs("outputs/model_comparison", exist_ok=True)
results_df.to_csv(
    "outputs/model_comparison/random_forest_tuning_comparison.csv",
    index=False,
)
print("\nSaved comparison to outputs/model_comparison/random_forest_tuning_comparison.csv")
