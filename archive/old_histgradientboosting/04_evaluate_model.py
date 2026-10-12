import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"])
y = train["condition"]

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
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ))
    ]), categorical_cols)
])

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_val)

print("Validation accuracy:", round(accuracy_score(y_val, pred), 4))
print("Validation macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
print("\nClassification report:")
print(classification_report(y_val, pred, digits=4))

cm = confusion_matrix(y_val, pred, labels=[0, 1, 2])
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=[0, 1, 2], yticklabels=[0, 1, 2]
)
plt.xlabel("Predicted condition")
plt.ylabel("Actual condition")
plt.title("HistGradientBoosting Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved plot to confusion_matrix.png")
