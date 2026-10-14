# Industrial Machine Condition Classification

## 1. Problem and Dataset

The objective of this project is to classify industrial machine operating conditions using sensor measurements, inspired by predictive maintenance applications.

This is a three-class classification problem:

- **Class 0 — Normal**
- **Class 1 — Thermal Failure**
- **Class 2 — Mechanical Failure**

The dataset consists of `student.csv` (training data) and a separate test dataset.

| Property | Training | Test |
|---|---:|---:|
| Samples | 16,000 | 4,000 |
| Columns | 13 | 12 |
| Target (`condition`) | Available | Not available |

The dataset includes ten numerical sensor features, one categorical feature (`unit_type`), and a machine identifier (`unit_id`). The identifier is excluded from model training.

## 2. Exploratory Data Analysis (EDA)

EDA examined the target distribution, missing values, sensor feature distributions, correlations, and relationships between sensor measurements and machine conditions.

Key observations:
- The Normal class accounts for 50% of samples, while each failure class accounts for 25%.
- Missing values occur in numerical and categorical features but affect less than 0.5% of observations per feature.
- Thermal measurements help distinguish thermal failures, while wear, vibration, and mechanical load provide information about mechanical failures.
- Several operating and thermal measurements are correlated.

## 3. Preprocessing

- Excluded `unit_id` from the predictor set.
- Imputed numerical missing values using the training-set median.
- Imputed missing `unit_type` values using the most frequent category.
- Applied one-hot encoding to `unit_type`.
- Used an 80:20 stratified train-validation split with `random_state=42`.
- Used scikit-learn pipelines and a `ColumnTransformer` to maintain consistent preprocessing.

## 4. Models and Experiments

The project is organized into two notebooks:

### `industrial_machine_classification.ipynb`
Contains dataset exploration, EDA, preprocessing, and initial classification experiments:
- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest

### `boosting.ipynb`
Contains experiments with boosting-based models, hyperparameter tuning, and ensemble methods:
- XGBoost
- CatBoost
- LightGBM
- XGBoost and CatBoost ensemble

## 5. Model Performance

The following results were obtained on the validation dataset.

| Model | Validation Accuracy | Macro F1-score |
|---|---:|---:|
| Gaussian Naive Bayes (Baseline) | 76.50% | 73.57% |
| Gaussian Naive Bayes (Tuned) | 77.53% | 74.41% |
| Logistic Regression (Tuned) | 82.16% | 80.35% |
| Decision Tree (Tuned) | 84.66% | 82.81% |
| Decision Tree (Refined) | 85.75% | 84.06% |
| Random Forest (Tuned) | 88.41% | 87.26% |
| LightGBM | 88.78% | 87.72% |
| CatBoost | 89.12% | 88.10% |
| XGBoost (Tuned) | 89.22% | 88.16% |
| XGBoost + CatBoost Ensemble | 89.31% | 88.31% |
| **XGBoost (Base)** | **89.34%** | **88.32%** |

Among the reported experiments, **XGBoost (Base)** achieved the highest validation accuracy and macro F1-score.

## 6. Technologies Used

- Python
- Jupyter Notebook
- pandas and NumPy
- scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Matplotlib and Seaborn

## 7. Test Predictions

The trained models can be used to generate predictions for the 4,000 unseen test samples and save the results as CSV files for submission.
