import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"])
y = train["condition"]

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

configs = [
    {"max_iter": 150, "learning_rate": 0.05,
     "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"max_iter": 200, "learning_rate": 0.08,
     "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"max_iter": 300, "learning_rate": 0.05,
     "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"max_iter": 200, "learning_rate": 0.08,
     "max_leaf_nodes": 15, "l2_regularization": 3.0},
    {"max_iter": 300, "learning_rate": 0.05,
     "max_leaf_nodes": 63, "l2_regularization": 3.0},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for i, params in enumerate(configs, start=1):
    model = HistGradientBoostingClassifier(
        **params, random_state=42
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1
    )

    row = {
        "config": i,
        "accuracy": scores["test_accuracy"].mean(),
        "macro_f1": scores["test_f1_macro"].mean(),
        "f1_std": scores["test_f1_macro"].std(),
        **params
    }
    results.append(row)

    print(
        f"Config {i}: accuracy={row['accuracy']:.4f}, "
        f"macro F1={row['macro_f1']:.4f}, "
        f"F1 std={row['f1_std']:.4f}"
    )

results = pd.DataFrame(results).sort_values(
    "macro_f1", ascending=False
)
results.to_csv("tuning_results.csv", index=False)

print("\nRanked results:")
print(results.to_string(index=False))
print("\nSaved to tuning_results.csv")
