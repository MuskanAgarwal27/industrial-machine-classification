from common import *
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Boosting libraries handle missing values natively -> no imputer needed
run_cv("RandomForest 500", lambda: make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)))

run_cv("LightGBM", lambda: lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=SEED, verbose=-1))

run_cv("XGBoost", lambda: xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED))

run_cv("CatBoost", lambda: CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=6,
    random_seed=SEED, verbose=0, thread_count=-1))
