# Industrial Machine Condition Classification

## 1. Objective
Build a supervised machine-learning classifier to predict the condition
of an industrial machine from its sensor measurements.

## 2. Dataset
- Training data: 16,000 rows and 13 columns.
- Test data: 4,000 rows and 12 columns.
- Target: condition, with classes 0, 1, and 2.
- Training target counts: class 0 = 8,000; class 1 = 4,000;
  class 2 = 4,000.
- unit_id is an identifier, not a predictive feature.

## 3. Data Preparation
- Split the labelled data into 80% training and 20% validation data.
- Used stratification to preserve class proportions.
- Imputed missing numeric values using the median.
- Imputed missing categorical values using the most frequent category.
- One-hot encoded unit_type.
- Used a fixed random seed for reproducibility.

## 4. Model
Random Forest classifier:
- Number of trees: 300
- Minimum samples per leaf: 2
- Maximum features per split: sqrt
- Random state: 42

Random Forest combines predictions from multiple decision trees.
Each tree is trained using a sample of the training data and a random
subset of features. Combining trees helps reduce the variance of a
single decision tree.

## 5. Evaluation
On the stratified 80/20 held-out validation split:

- Accuracy: 0.8869
- Macro F1-score: 0.8758

### Per-class results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.8942 | 0.9406 | 0.9168 | 1600 |
| 1 | 0.8732 | 0.8263 | 0.8491 | 800 |
| 2 | 0.8842 | 0.8400 | 0.8615 | 800 |

### Confusion matrix
Rows are actual classes; columns are predicted classes.

| Actual / Predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 1505 | 56 | 39 |
| 1 | 90 | 661 | 49 |
| 2 | 88 | 40 | 672 |

## 6. Interpretation
The model performs best on class 0. Classes 1 and 2 are more difficult
to distinguish. Some examples from both classes are incorrectly
predicted as class 0. Macro F1 is included because it gives equal
importance to each class rather than allowing the largest class to
dominate the evaluation.

## 7. Limitations
- Validation performance is an estimate and may differ from performance
  on unseen test data.
- The dataset contains missing feature values, which are imputed.
- Correlated sensor measurements may contain overlapping information.
- Predictions should not be interpreted as causal explanations.

## 8. Reproducibility and Artifacts
The project scripts record exploratory analysis, feature experiments,
model comparisons, validation metrics, and test prediction generation.
Random seeds are fixed where applicable.

The local prediction file is submission.csv. It has 4,000 rows and the
required columns unit_id and condition. It has not been submitted.

## 9. Final Status
Main model: Random Forest.
Held-out validation accuracy: 88.69%.
Held-out validation macro F1: 0.8758.
Kaggle submission: not performed.
