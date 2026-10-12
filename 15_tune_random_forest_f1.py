import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

params = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 15, 20, 25, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": ["sqrt", 0.5, 0.7, 0.9],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=params,
    n_iter=30,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
    refit=True,
)

print("Tuning Random Forest for Macro F1...")
search.fit(X_train, y_train)

print("\nBest CV Macro F1:", round(search.best_score_, 5))
print("Best parameters:")
for key, value in search.best_params_.items():
    print(f"{key}: {value}")

# Evaluate the best candidate on the untouched validation split
pred = search.best_estimator_.predict(X_valid)
print("\nValidation accuracy:", round(accuracy_score(y_valid, pred), 5))
print("Validation Macro F1:", round(f1_score(y_valid, pred, average="macro"), 5))

# Save experiment results without overwriting existing submissions
os.makedirs("outputs/model_comparison", exist_ok=True)
pd.DataFrame([{
    "cv_macro_f1": search.best_score_,
    "validation_accuracy": accuracy_score(y_valid, pred),
    "validation_macro_f1": f1_score(y_valid, pred, average="macro"),
    "best_params": str(search.best_params_),
}]).to_csv(
    "outputs/model_comparison/random_forest_f1_tuning.csv",
    index=False,
)

print("\nSaved metrics to outputs/model_comparison/random_forest_f1_tuning.csv")
