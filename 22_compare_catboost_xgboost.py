import os
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Load data and create the same validation split
# --------------------------------------------------
DATA_PATH = "data/student_dataset.csv"
OUTPUT_DIR = "outputs/model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [
    col for col in X_train.columns if col not in numeric_cols
]

print("Dataset shape:", df.shape)
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)


# --------------------------------------------------
# 2. Prepare CatBoost data
# CatBoost can handle categorical features natively.
# --------------------------------------------------
X_train_cb = X_train.copy()
X_val_cb = X_val.copy()

for col in categorical_cols:
    X_train_cb[col] = X_train_cb[col].fillna("Missing").astype(str)
    X_val_cb[col] = X_val_cb[col].fillna("Missing").astype(str)

cat_indices = [
    X_train_cb.columns.get_loc(col) for col in categorical_cols
]


# --------------------------------------------------
# 3. Prepare XGBoost preprocessing
# XGBoost uses imputation and one-hot encoding here.
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])


# --------------------------------------------------
# 4. Train and evaluate both models
# --------------------------------------------------
models = [
    (
        "CatBoost",
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        ),
        X_train_cb,
        X_val_cb,
        cat_indices,
    ),
    (
        "XGBoost",
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                device="cpu",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        X_train,
        X_val,
        None,
    ),
]

results = []

for name, model, train_X, val_X, categorical_features in models:
    print(f"\n{'=' * 55}")
    print("Training:", name)

    start = time.time()

    if name == "CatBoost":
        model.fit(
            train_X,
            y_train,
            cat_features=categorical_features,
        )
    else:
        model.fit(train_X, y_train)

    predictions = model.predict(val_X)

    # Normalize predictions to a 1-D array.
    predictions = np.asarray(predictions).reshape(-1)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )

    elapsed = time.time() - start

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_and_prediction_seconds": round(elapsed, 2),
    })

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")
    print(f"Time: {elapsed:.2f} seconds")
    print("\nClassification report:")
    print(classification_report(y_val, predictions, zero_division=0))


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("macro_f1", ascending=False)

output_path = os.path.join(
    OUTPUT_DIR, "catboost_xgboost_comparison.csv"
)
results_df.to_csv(output_path, index=False)

print("\nFinal comparison:")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {output_path}")
