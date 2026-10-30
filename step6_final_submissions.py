from common import *
import os
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier

OUT = "outputs/submissions_v2"
os.makedirs(OUT, exist_ok=True)
sample = pd.read_csv(DATA + "sample_submission.csv")

svm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    SVC(C=30, gamma=0.05, kernel="rbf", probability=True, random_state=SEED))
lr3 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    PolynomialFeatures(degree=3), StandardScaler(),
                    LogisticRegression(C=0.3, max_iter=5000))
cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                         random_seed=SEED, verbose=0, thread_count=-1)

probs = {}
for name, model in [("svm", svm), ("logreg_poly3", lr3), ("catboost", cat)]:
    print("Training", name, "on all 16,000 rows...")
    model.fit(X, y)
    probs[name] = model.predict_proba(X_test)

def save(name, p):
    sub = pd.DataFrame({"unit_id": test["unit_id"], "condition": p.argmax(1).astype(int)})
    assert sub["unit_id"].equals(sample["unit_id"]) and len(sub) == 4000
    sub.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"Saved {name}.csv  class mix:", sub.condition.value_counts(normalize=True).sort_index().round(3).tolist())
    return sub

b = save("blend_svm_lr3_catboost", (probs["svm"] + probs["logreg_poly3"] + probs["catboost"]) / 3)
s = save("svm", probs["svm"])
l = save("logreg_poly3", probs["logreg_poly3"])

print("\nAgreement blend vs svm:", (b.condition == s.condition).mean().round(4),
      "| blend vs lr3:", (b.condition == l.condition).mean().round(4))
