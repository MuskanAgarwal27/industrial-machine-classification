import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
DATA = "data/"

train = pd.read_csv(DATA + "student_dataset.csv")
test  = pd.read_csv(DATA + "test.csv")
y = train["condition"].values

def make_X(df):
    """Numeric features + one-hot unit_type (NaN kept as its own column)."""
    X = df.drop(columns=["unit_id", "condition"], errors="ignore")
    dummies = pd.get_dummies(X["unit_type"], prefix="type", dummy_na=True).astype(float)
    return pd.concat([X.drop(columns="unit_type"), dummies], axis=1)

X      = make_X(train)
X_test = make_X(test).reindex(columns=X.columns, fill_value=0)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(name, make_model, X=X, y=y):
    """5-fold CV. make_model() must return a fresh unfitted model/pipeline.
    Returns out-of-fold probabilities (needed later for blending)."""
    oof = np.zeros((len(X), 3))
    accs = []
    for tr_idx, va_idx in CV.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])
        accs.append(accuracy_score(y[va_idx], oof[va_idx].argmax(1)))
    pred = oof.argmax(1)
    print(f"{name:35s} acc {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
          f"macro-F1 {f1_score(y, pred, average='macro'):.4f}")
    np.save(f"oof_{name.replace(' ', '_')}.npy", oof)
    return oof
