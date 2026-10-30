import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from common import *

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def svm(C, gamma):
    return lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),                 # SVM needs scaled features
        SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=SEED),
    )

for C, g in [(3, 0.05), (10, 0.05), (30, 0.05), (10, 0.1)]:
    run_cv(f"SVM C={C} gamma={g}", svm(C, g))
