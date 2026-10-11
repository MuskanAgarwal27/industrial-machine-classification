
import pandas as pd

from sklearn.model_selection import train_test_split
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
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/student_dataset.csv")

X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# Same preprocessing for all models
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Transform once. Dense data is needed by GaussianNB.
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42,
    ),
}

results = []

for name, model in models.items():
    print(f"Training {name}...", flush=True)

    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_val_processed)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Macro F1", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

output_path = "outputs/model_comparison"
import os
os.makedirs(output_path, exist_ok=True)

results_df.to_csv(
    f"{output_path}/model_comparison.csv",
    index=False,
)

print(f"\nResults saved to {output_path}/model_comparison.csv")
