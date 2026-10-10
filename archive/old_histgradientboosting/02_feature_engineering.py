import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

train = pd.read_csv("data/student_dataset.csv")
X = train.drop(columns=["condition", "unit_id"])
y = train["condition"]

def engineer(data):
    data = data.copy()
    data["temperature_difference"] = (
        data["process_temp"] - data["ambient_temp"]
    )
    data["torque_speed_product"] = (
        data["drive_torque"] * data["shaft_speed"]
    )
    return data

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ))
    ]), categorical_cols)
])

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, features in [
    ("Baseline", X),
    ("Engineered", engineer(X))
]:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    scores = cross_validate(
        pipeline, features, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1
    )
    print(f"\n{name}")
    print("Fold accuracies:",
          np.round(scores["test_accuracy"], 4))
    print("Mean accuracy:",
          round(scores["test_accuracy"].mean(), 4))
    print("Mean macro F1:",
          round(scores["test_f1_macro"].mean(), 4))
    print("Macro F1 standard deviation:",
          round(scores["test_f1_macro"].std(), 4))
