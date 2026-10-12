exec(open("train.py").read().split("for name, model in models.items():")[0])

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone

# HistGradientBoosting does not directly accept
# the sparse output produced by OneHotEncoder.
# Use dense one-hot encoded output for this model.
preprocessor.set_params(
    cat__encoder__sparse_output=False
)

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

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_val)

print("Accuracy:", round(accuracy_score(y_val, pred), 4))
print("Macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
print(classification_report(y_val, pred, zero_division=0))
