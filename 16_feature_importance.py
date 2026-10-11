import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

# Load data
train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]

# Same validation split used in our previous comparison
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
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

# Original baseline Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

print("Training baseline Random Forest...")
model.fit(X_train, y_train)

baseline_f1 = f1_score(
    y_valid,
    model.predict(X_valid),
    average="macro",
)
print(f"Validation Macro F1: {baseline_f1:.5f}")

print("\nCalculating permutation importance...")
importance = permutation_importance(
    model,
    X_valid,
    y_valid,
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)

results = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False)

os.makedirs("outputs/evaluation/random_forest", exist_ok=True)
output_path = "outputs/evaluation/random_forest/feature_importance.csv"
results.to_csv(output_path, index=False)

print("\nTop features:")
print(results.head(15).to_string(index=False))
print(f"\nSaved all results to {output_path}")
