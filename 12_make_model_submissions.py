
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

# Load labelled training data and test data
train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

# Separate features and target
X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]

# Preserve the same feature columns and order in test data
X_test = test.drop(columns=["unit_id"], errors="ignore")
X_test = X_test.reindex(columns=X.columns)

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Models
models = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        random_state=42,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "logistic_regression": LogisticRegression(max_iter=2000),
    "naive_bayes": GaussianNB(),
}

output_dir = "outputs/model_submissions"
os.makedirs(output_dir, exist_ok=True)

for name, classifier in models.items():
    print(f"Training {name}...", flush=True)

    # Fit preprocessing on all labelled training rows
    model_preprocessor = preprocessor
    X_train_processed = model_preprocessor.fit_transform(X, y)
    X_test_processed = model_preprocessor.transform(X_test)

    # GaussianNB needs dense arrays
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    classifier.fit(X_train_processed, y)
    predictions = classifier.predict(X_test_processed)

    # Start with the competition's sample submission
    submission = sample.copy()
    submission["condition"] = predictions

    # Check submission structure
    assert len(submission) == len(test)
    assert submission["condition"].notna().all()
    assert set(submission["condition"].unique()).issubset(set(y.unique()))

    output_file = f"{output_dir}/submission_{name}.csv"
    submission.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nAll five submission files created successfully.")
