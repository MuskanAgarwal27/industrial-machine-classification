# Industrial Machine Condition Classification

## Overview
This project predicts industrial machine condition classes from sensor
measurements using supervised machine learning.

The final selected model is a Random Forest classifier. HistGradientBoosting
and Extra Trees were explored during experimentation, but Random Forest was
selected for the final workflow to comply with the course model restrictions.

## Dataset
- Training data: `data/student_dataset.csv` (16,000 rows)
- Test data: `data/test.csv` (4,000 rows)
- Sample submission format: `data/sample_submission.csv`
- Target column: `condition`
- Identifier column: `unit_id`

## Method
1. Inspect the dataset and check data consistency.
2. Impute missing numerical values using the median.
3. Impute missing categorical values using the most frequent value.
4. One-hot encode categorical features.
5. Train and evaluate a Random Forest classifier.
6. Generate predictions for the unlabeled test dataset.

The identifier `unit_id` is retained for output but excluded from model
features.

## Final model
- Algorithm: RandomForestClassifier
- Number of trees: 300
- Minimum samples per leaf: 2
- Maximum features: `sqrt`
- Random state: 42
- Class weighting: default (not balanced)

## Validation results
On a stratified 80/20 holdout split:
- Accuracy: 0.8869
- Macro F1-score: 0.8758

These are local validation results, not competition leaderboard scores.

## Project files
- `01_eda.py`: exploratory data analysis
- `07_data_checks.py`: dataset consistency checks
- `08_make_predictions.py`: train the final Random Forest on all labeled
  rows and create local predictions
- `09_evaluate_random_forest.py`: holdout evaluation and confusion matrix
- `PROJECT_REPORT.md`: detailed report
- `requirements.txt`: Python dependencies
- `data/`: supplied datasets and sample submission
- `archive/old_histgradientboosting/`: archived HGB scripts and results
- `archive/experiments/`: archived mixed-model comparison

## Setup
Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run
From the project root, run:

```bash
python 07_data_checks.py
python 01_eda.py
python 09_evaluate_random_forest.py
python 08_make_predictions.py
```

Some scripts may take time to finish because model training is required.

The prediction script writes `submission.csv` locally. It does not upload
or submit anything to Kaggle.

## Important
The test set has no target labels, so its predictive performance cannot be
measured locally without ground truth. Holdout validation provides the
reported performance estimate.
