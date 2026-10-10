import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

X_train = train.drop(columns=["condition", "unit_id"])
y_train = train["condition"]
X_test = test.drop(columns=["unit_id"])

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

output = pd.DataFrame({
    "unit_id": test["unit_id"],
    "condition": predictions
})

assert len(output) == len(test) == len(sample)
assert output.columns.tolist() == sample.columns.tolist()
assert output["unit_id"].equals(sample["unit_id"])
assert output["condition"].notna().all()
assert set(output["condition"].unique()).issubset(set(y_train.unique()))

output.to_csv("submission.csv", index=False)

print("Local Random Forest prediction file created.")
print("Rows:", len(output))
print("\nPredicted class counts:")
print(output["condition"].value_counts().sort_index())
print("\nFirst five rows:")
print(output.head())
print("\nAll checks passed. Nothing was submitted to Kaggle.")
