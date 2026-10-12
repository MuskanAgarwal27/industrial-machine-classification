import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Load the full training data and competition test data
train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")

# Separate features and target
X = train.drop(columns=["condition", "unit_id"])
y = train["condition"]
X_test = test.drop(columns=["unit_id"])

# Reuse the preprocessing configuration from train.py
exec(open("train.py").read().split("for name, model in models.items():")[0])

preprocessor.set_params(cat__encoder__sparse_output=False)

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", model)
])

# Train on ALL labeled training rows
pipeline.fit(X, y)

# Predict the competition test rows
predictions = pipeline.predict(X_test)

# Create the submission in the required format
submission = pd.DataFrame({
    "unit_id": test["unit_id"],
    "condition": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission created: submission.csv")
print(submission.head())
print("Shape:", submission.shape)
print("Predicted class counts:")
print(submission["condition"].value_counts().sort_index())
