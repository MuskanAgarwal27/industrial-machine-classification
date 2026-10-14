import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = "data"
OUTPUT_DIR = "outputs/submissions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(f"{DATA_DIR}/student_dataset.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

target = "condition"
id_col = "unit_id"

# Keep test IDs and order exactly as given in the sample submission.
if not sample_df[id_col].equals(test_df[id_col]):
    raise ValueError(
        "unit_id values/order in test.csv do not match sample_submission.csv"
    )

feature_cols = [
    col for col in train_df.columns
    if col not in [target, id_col]
]

X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [
    col for col in feature_cols if col not in numeric_cols
]

# CatBoost: native categorical-feature support.
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()

for col in categorical_cols:
    X_train_cb[col] = X_train_cb[col].fillna("Missing").astype(str)
    X_test_cb[col] = X_test_cb[col].fillna("Missing").astype(str)

cat_indices = [
    X_train_cb.columns.get_loc(col) for col in categorical_cols
]

cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)

print("Training CatBoost on all labeled training rows...")
cat_model.fit(X_train_cb, y_train, cat_features=cat_indices)
cat_predictions = np.asarray(
    cat_model.predict(X_test_cb)
).reshape(-1).astype(int)

cat_submission = pd.DataFrame({
    id_col: test_df[id_col],
    target: cat_predictions,
})

cat_path = f"{OUTPUT_DIR}/submission_catboost.csv"
cat_submission.to_csv(cat_path, index=False)
print(f"Saved: {cat_path}")

# XGBoost: same imputation and one-hot encoding as the comparison script.
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

xgb_model = Pipeline([
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
])

print("Training XGBoost on all labeled training rows...")
xgb_model.fit(X_train, y_train)
xgb_predictions = np.asarray(
    xgb_model.predict(X_test)
).reshape(-1).astype(int)

xgb_submission = pd.DataFrame({
    id_col: test_df[id_col],
    target: xgb_predictions,
})

xgb_path = f"{OUTPUT_DIR}/submission_xgboost.csv"
xgb_submission.to_csv(xgb_path, index=False)
print(f"Saved: {xgb_path}")

# Validate both output files.
for path in [cat_path, xgb_path]:
    result = pd.read_csv(path)

    assert list(result.columns) == [id_col, target]
    assert len(result) == len(sample_df)
    assert result[id_col].equals(sample_df[id_col])
    assert result[target].notna().all()
    assert set(result[target].unique()).issubset(set(y_train.unique()))

    print(f"\nVerified: {path}")
    print(f"Rows: {len(result)}")
    print(f"Predicted classes: {sorted(result[target].unique())}")
    print(result.head())

print("\nBoth submission files were created and validated.")
