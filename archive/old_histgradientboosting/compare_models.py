exec(open("train.py").read().split("for name, model in models.items():")[0])

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.base import clone

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_val, pred), 4))
    print("Macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
    print(classification_report(y_val, pred, zero_division=0))
