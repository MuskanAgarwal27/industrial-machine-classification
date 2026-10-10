import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.model_selection import StratifiedKFold, cross_validate

df = pd.read_csv("data/student_dataset.csv")
X = df.drop(columns=["condition", "unit_id"])
y = df["condition"]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ))
        ]), categorical_cols)
    ])

models = {
    "HGB tuned": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05,
        max_leaf_nodes=15, l2_regularization=1.0,
        random_state=42
    ),
    "HGB balanced": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05,
        max_leaf_nodes=15, l2_regularization=1.0,
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1
    ),
    "Random Forest balanced": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1
    ),
    "Extra Trees balanced": ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])

    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1
    )

    row = {
        "model": name,
        "accuracy_mean": scores["test_accuracy"].mean(),
        "macro_f1_mean": scores["test_f1_macro"].mean(),
        "macro_f1_std": scores["test_f1_macro"].std()
    }
    results.append(row)

    print(
        f"{name}: accuracy={row['accuracy_mean']:.4f}, "
        f"macro F1={row['macro_f1_mean']:.4f}, "
        f"F1 std={row['macro_f1_std']:.4f}"
    )

results = pd.DataFrame(results).sort_values(
    "macro_f1_mean", ascending=False
)
results.to_csv("model_comparison.csv", index=False)

print("\nRanked results:")
print(results.to_string(index=False))
print("\nSaved to model_comparison.csv")
