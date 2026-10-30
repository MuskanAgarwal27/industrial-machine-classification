from common import *
from itertools import combinations

models = {
    "SVM":      "oof_SVM_C=30_gamma=0.05.npy",
    "LR3":      "oof_LogReg_poly3_C=0.3.npy",
    "LR2":      "oof_LogReg_poly2_C=0.3.npy",
    "XGB":      "oof_XGBoost.npy",
    "CatBoost": "oof_CatBoost.npy",
    "LGBM":     "oof_LightGBM.npy",
    "RF":       "oof_RandomForest_500.npy",
}
P = {k: np.load(f) for k, f in models.items()}

for k, p in P.items():
    print(f"{k:10s} alone: {accuracy_score(y, p.argmax(1)):.4f}")

print("\nTop blends (equal-weight average of probabilities):")
results = []
for r in [2, 3, 4]:
    for combo in combinations(P, r):
        avg = sum(P[k] for k in combo) / r
        results.append((accuracy_score(y, avg.argmax(1)), combo))
for acc, combo in sorted(results, reverse=True)[:10]:
    print(f"  {acc:.4f}  {' + '.join(combo)}")
