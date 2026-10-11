import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]
X_test = test.drop(columns=["unit_id"], errors="ignore")

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=15,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )),
])

print("Training HistGradientBoosting on all labeled data...")
model.fit(X, y)

predictions = model.predict(X_test)

# Preserve the exact column names and order required by sample_submission.csv.
submission = sample.copy()
if len(submission) != len(predictions):
    raise ValueError("Test predictions do not match sample submission row count.")

submission["condition"] = predictions

output_dir = "outputs/submissions"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(
    output_dir, "submission_hist_gradient_boosting_tuned.csv"
)
submission.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Rows: {len(submission)}")
print("\nPredicted class counts:")
print(submission["condition"].value_counts().sort_index())
print("\nFirst five rows:")
print(submission.head())
