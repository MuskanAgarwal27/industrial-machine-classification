# Industrial Machine Condition Classification

## Overview

A supervised machine learning project that predicts industrial machine
condition classes from sensor measurements.

Random Forest is the selected final model to comply with the course model
restrictions. Other algorithms are used for comparison and experimentation.

The project explores five classification algorithms, compares their
validation performance, and generates Kaggle submission files.

## Dataset

- Training data: `data/student_dataset.csv` (16,000 rows)
- Test data: `data/test.csv` (4,000 rows)
- Submission template: `data/sample_submission.csv`
- Target: `condition`
- Identifier: `unit_id`

The identifier is retained in submission files but excluded from model
features.

## Workflow

1. Explore distributions, missing values, correlations and outliers.
2. Validate dataset consistency.
3. Impute missing numerical and categorical values.
4. One-hot encode categorical features.
5. Train and compare classification models.
6. Evaluate models on a stratified validation split.
7. Tune promising models and generate predictions for the test set.

## Models compared

| Model | Validation accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 0.8872 | 0.8762 |
| HistGradientBoosting | 0.8866 | 0.8757 |
| Extra Trees | 0.8781 | 0.8643 |
| Logistic Regression | 0.8225 | 0.8043 |
| Gaussian Naive Bayes | 0.7650 | 0.7351 |

The comparison uses an 80/20 stratified split with random state 42.
Scores are local validation results, not leaderboard scores.

Random Forest is the current leading model by validation Macro F1.
Its competition leaderboard score has reached 0.88750.

## Main scripts

- `01_eda.py`: exploratory data analysis.
- `07_data_checks.py`: dataset consistency checks.
- `08_make_predictions.py`: generate the original Random Forest submission.
- `09_evaluate_random_forest.py`: evaluate Random Forest and produce metrics.
- `10_naive_bayes.py`: evaluate Gaussian Naive Bayes.
- `11_compare_models.py`: compare five classification models.
- `12_make_model_submissions.py`: generate one submission per model.
- `13_tune_random_forest.py`: tune Random Forest hyperparameters.
- `PROJECT_REPORT.md`: project report.
- `requirements.txt`: Python dependencies.

## Output folders

- `outputs/eda/`: exploratory analysis plots and tables.
- `outputs/random_forest/`: Random Forest evaluation results.
- `outputs/model_comparison/`: model comparison table.
- `outputs/model_submissions/`: locally generated Kaggle submission files;
  excluded from Git.

## Setup

From the project root:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## Run the workflow

    python 07_data_checks.py
    python 01_eda.py
    python 09_evaluate_random_forest.py
    python 11_compare_models.py
    python 12_make_model_submissions.py

For hyperparameter tuning:

    python 13_tune_random_forest.py

Some scripts train multiple models and may take several minutes.

## Submission format

Generated CSVs contain the competition's required `unit_id` and
`condition` columns. Inspect the sample submission and verify the output
before uploading it to Kaggle.

The prediction scripts do not upload anything to Kaggle automatically.

## Limitations

The test labels are unavailable locally, so local test accuracy cannot
be measured. Validation and leaderboard results may differ.

The tuned Random Forest's cross-validation score is stored separately
from the holdout comparison scores and should not be directly compared
as though both came from the same split.
