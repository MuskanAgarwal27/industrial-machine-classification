from common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def poly_lr(degree, C):
    return lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),                       # scale BEFORE expanding
        PolynomialFeatures(degree=degree),      # squares + interactions (+ cubes)
        StandardScaler(),                       # rescale the new features
        LogisticRegression(C=C, max_iter=5000),
    )

run_cv("LogReg poly2 C=0.3", poly_lr(2, 0.3))
run_cv("LogReg poly3 C=0.03", poly_lr(3, 0.03))
run_cv("LogReg poly3 C=0.1", poly_lr(3, 0.1))
run_cv("LogReg poly3 C=0.3", poly_lr(3, 0.3))
