import numpy as np
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score

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
probs = pipeline.predict_proba(X_val)

print("Accuracy:", round(accuracy_score(y_val, pred), 4))
print("Macro F1:", round(f1_score(y_val, pred, average="macro"), 4))

# Save individual mistakes for later inspection
errors = X_val.copy()
errors["actual_condition"] = y_val
errors["predicted_condition"] = pred
errors["prediction_confidence"] = probs.max(axis=1)
errors = errors[errors["actual_condition"] != errors["predicted_condition"]]
errors.to_csv("misclassified_examples.csv", index=True)

print("\nMisclassified examples:", len(errors))
print("Most common actual -> predicted errors:")
print(
    errors.groupby(
        ["actual_condition", "predicted_condition"]
    ).size().sort_values(ascending=False).head(10)
)

# Permutation importance using macro F1
importance = permutation_importance(
    pipeline, X_val, y_val,
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv("permutation_importance.csv", index=False)

print("\nPermutation importance (macro F1):")
print(importance_df.round(5).to_string(index=False))

plt.figure(figsize=(9, 6))
plot_data = importance_df.sort_values("importance_mean")
plt.barh(
    plot_data["feature"],
    plot_data["importance_mean"],
    xerr=plot_data["importance_std"]
)
plt.xlabel("Decrease in macro F1 after shuffling")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150)
plt.close()

print("\nSaved:")
print("- misclassified_examples.csv")
print("- permutation_importance.csv")
print("- permutation_importance.png")
