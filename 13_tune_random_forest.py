
import os
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

X = train.drop(columns=["condition", "unit_id"], errors="ignore")
y = train["condition"]
X_test = test.reindex(columns=X.columns)

# Preprocessing
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )),
])

# Try different Random Forest settings
param_distributions = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 10, 15, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": ["sqrt", 0.7, 1.0],
    "model__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
    refit=True,
)

print("Searching for better Random Forest settings...")
search.fit(X, y)

print("\nBest cross-validation accuracy:", round(search.best_score_, 5))
print("Best parameters:")
for key, value in search.best_params_.items():
    print(f"{key}: {value}")

# The best pipeline has now been refitted on all labelled data.
predictions = search.best_estimator_.predict(X_test)

submission = sample.copy()
submission["condition"] = predictions

assert len(submission) == len(test)
assert submission["condition"].notna().all()
assert set(submission["condition"].unique()).issubset(set(y.unique()))

output_dir = "outputs/submissions"
os.makedirs(output_dir, exist_ok=True)

output_file = f"{output_dir}/submission_random_forest_tuned.csv"
submission.to_csv(output_file, index=False)

print(f"\nNew submission saved to: {output_file}")
