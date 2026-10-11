
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load labelled training data
df = pd.read_csv("data/student_dataset.csv")

# 2. Separate features and target
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
y = df["condition"]

# 3. Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# 4. Preprocess numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# 5. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB()),
])

# GaussianNB requires dense input, so convert the transformed
# feature matrix to a dense array.
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_val_processed = preprocessor.transform(X_val)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_val_processed = X_val_processed.toarray()

# 7. Train and predict
classifier = GaussianNB()
classifier.fit(X_train_processed, y_train)
predictions = classifier.predict(X_val_processed)

# 8. Evaluate
print("Naive Bayes Results")
print("-------------------")
print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
print(f"Macro F1: {f1_score(y_val, predictions, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_val, predictions))
