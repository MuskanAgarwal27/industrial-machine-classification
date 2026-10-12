# Industrial Machine Condition Classification

## 1. Introduction

This project focuses on classifying industrial machine conditions using sensor measurements and machine-related features. It is a multiclass classification problem with three target classes:

- **0 — Normal**
- **1 — Thermal Failure**
- **2 — Mechanical Failure**

The objective is to develop and compare machine learning models that can accurately identify the operating condition of a machine.

## 2. Exploratory Data Analysis (EDA)

The dataset (`student.csv`) contains 11 sensor features, `unit_type`, `unit_id`, and the target variable `condition`.

The EDA phase involved:

- Examining the dataset structure, feature types, and summary statistics.
- Investigating the distribution of the target classes.
- Inspecting sensor feature distributions and potential outliers.
- Checking for missing values.
- Exploring relationships between sensor measurements and machine conditions.

These analyses helped understand the dataset and identify relevant preprocessing requirements before model training.

## 3. Preprocessing

The preprocessing workflow focused on preparing the features for classification:

- Handling missing values where required.
- Encoding categorical features such as `unit_type`.
- Excluding `unit_id` as an identifier rather than a predictive feature.
- Separating input features and the target variable.
- Using training and validation data to evaluate model performance and guide model selection.

## 4. Models Experimented With

The following models were evaluated progressively, starting with a simple probabilistic baseline and moving towards more flexible classification models.

1. **Gaussian Naive Bayes — Baseline:** Established an initial performance benchmark using Gaussian feature distributions.
2. **Tuned Gaussian Naive Bayes:** Adjusted `var_smoothing` to improve classification performance.
3. **Tuned Logistic Regression:** Evaluated a linear classification approach with hyperparameter tuning.
4. **Tuned Decision Tree:** Used a tree-based model to capture nonlinear relationships and tuned its hyperparameters.
5. **Refined Decision Tree:** Further refined the tree configuration to improve validation performance.
6. **Random Forest — Team Comparison:** Included as a comparison result obtained by a teammate; it was not part of my individual model experimentation.

## 5. Model Performance

The models were compared using validation accuracy. The results are summarized below.

| Model | Accuracy |
|---|---:|
| Gaussian Naive Bayes — Baseline | 76.56% |
| Tuned Gaussian Naive Bayes | 77.53% |
| Tuned Logistic Regression | 82.16% |
| Tuned Decision Tree | 84.66% |
| Refined Decision Tree | 85.75% |
| Random Forest — Teammate's Result | 88.41% |

Among the models experimented with individually, the **Refined Decision Tree achieved the highest validation accuracy of 85.75%**. The Random Forest result of 88.41% is included for team-level comparison.

## 6. Conclusion

The experiments show progressive improvements from the initial Naive Bayes baseline to the refined Decision Tree model. The results highlight the importance of model selection and hyperparameter tuning in industrial machine condition classification.


## 7. Tools and Libraries

- Python
- Jupyter Notebook / Google Colab
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn, if used for EDA visualizations
