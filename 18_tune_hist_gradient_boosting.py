import os
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/student_dataset.csv")
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingClassifier(random_state=42)),
])

param_distributions = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__max_iter": [100, 200, 300],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__max_depth": [None, 6, 10],
    "model__min_samples_leaf": [10, 20, 30, 50],
    "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Tuning HistGradientBoosting for Macro F1...", flush=True)
search.fit(X_train, y_train)

print(f"\nBest CV Macro F1: {search.best_score_:.5f}")
print("Best parameters:")
for key, value in search.best_params_.items():
    print(f"  {key}: {value}")

predictions = search.best_estimator_.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")

print(f"\nValidation accuracy: {accuracy:.5f}")
print(f"Validation Macro F1: {macro_f1:.5f}")
print("\nClassification report:")
print(classification_report(y_val, predictions, zero_division=0))

os.makedirs("outputs/model_comparison", exist_ok=True)
results = pd.DataFrame([{
    "model": "HistGradientBoosting tuned",
    "cv_macro_f1": search.best_score_,
    "validation_accuracy": accuracy,
    "validation_macro_f1": macro_f1,
    "best_params": str(search.best_params_),
}])

output_path = "outputs/model_comparison/hist_gradient_boosting_tuning.csv"
results.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
