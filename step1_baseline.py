from common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("Train:", X.shape, "Test:", X_test.shape)
print("Class counts:", np.bincount(y))

run_cv("LogReg plain", lambda: make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=2000),
))
