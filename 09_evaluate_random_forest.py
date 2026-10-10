import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

os.makedirs("outputs/random_forest", exist_ok=True)

df = pd.read_csv("data/student_dataset.csv")
X = df.drop(columns=["condition", "unit_id"])
y = df["condition"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_val)

accuracy = accuracy_score(y_val, pred)
macro_f1 = f1_score(y_val, pred, average="macro")
report = classification_report(y_val, pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_val, pred, labels=[0, 1, 2])

print("Validation accuracy:", round(accuracy, 4))
print("Validation macro F1:", round(macro_f1, 4))
print("\nClassification report:")
print(classification_report(y_val, pred, digits=4, zero_division=0))
print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)

with open("outputs/random_forest/metrics.json", "w") as f:
    json.dump({
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }, f, indent=2)

pd.DataFrame(
    cm, index=["actual_0", "actual_1", "actual_2"],
    columns=["predicted_0", "predicted_1", "predicted_2"]
).to_csv("outputs/random_forest/confusion_matrix.csv")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=[0, 1, 2], yticklabels=[0, 1, 2]
)
plt.xlabel("Predicted condition")
plt.ylabel("Actual condition")
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/random_forest/confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved evaluation files in outputs/random_forest/")
