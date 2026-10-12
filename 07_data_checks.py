import pandas as pd

train = pd.read_csv("data/student_dataset.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

print("TRAIN shape:", train.shape)
print("TEST shape:", test.shape)
print("SAMPLE SUBMISSION shape:", sample.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())

print("\nTarget distribution:")
print(train["condition"].value_counts().sort_index())

print("\nMissing values in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nMissing values in test:")
print(test.isnull().sum()[test.isnull().sum() > 0])

print("\nDuplicate train IDs:", train["unit_id"].duplicated().sum())
print("Duplicate test IDs:", test["unit_id"].duplicated().sum())

print("\nTrain/test feature columns match:",
      set(train.columns) - {"condition"} == set(test.columns))

print("Test IDs match sample submission:",
      test["unit_id"].reset_index(drop=True).equals(
          sample["unit_id"].reset_index(drop=True)
      ))

print("\nSample submission columns:", sample.columns.tolist())
print("Allowed target classes:", sorted(train["condition"].unique()))
