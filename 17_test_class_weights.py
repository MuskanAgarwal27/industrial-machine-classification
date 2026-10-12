import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/student_dataset.csv")
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

def make_model(class_weight):
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])

results = []

for weight in [None, "balanced", "balanced_subsample"]:
    print(f"\nTesting class_weight={weight}")
    model = make_model(weight)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(y_valid, predictions, average="macro")

    results.append({
        "class_weight": str(weight),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    print(f"Accuracy: {accuracy:.5f}")
    print(f"Macro F1: {macro_f1:.5f}")
    print(classification_report(y_valid, predictions, zero_division=0))

results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
os.makedirs("outputs/model_comparison", exist_ok=True)
results_df.to_csv(
    "outputs/model_comparison/class_weight_comparison.csv",
    index=False,
)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved to outputs/model_comparison/class_weight_comparison.csv")
