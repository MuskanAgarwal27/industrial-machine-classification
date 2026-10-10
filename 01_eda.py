from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
OUT_DIR = Path("outputs/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "student_dataset.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "condition"
ID = "unit_id"

numeric_cols = train.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [TARGET, ID]]
categorical_cols = train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != ID]

sns.set_theme(style="whitegrid")

print("\n========== DATA OVERVIEW ==========")
print("Training shape:", train.shape)
print("Test shape:", test.shape)
print("\nData types:\n", train.dtypes)
print("\nFirst five rows:\n", train.head())
print("\nTarget counts:\n", train[TARGET].value_counts().sort_index())
print("\nTarget proportions:\n",
      train[TARGET].value_counts(normalize=True).sort_index().round(4))

print("\n========== DATA QUALITY ==========")
print("Duplicate training rows:", train.duplicated().sum())
print("Duplicate training IDs:", train[ID].duplicated().sum())
print("Duplicate test IDs:", test[ID].duplicated().sum())
print("Missing target values:", train[TARGET].isna().sum())

missing = pd.DataFrame({
    "train_missing": train.isna().sum(),
    "train_missing_pct": (train.isna().mean() * 100).round(2),
})
test_aligned = test.reindex(columns=train.columns)
missing["test_missing"] = test_aligned.isna().sum()
missing["test_missing_pct"] = (
    test_aligned.isna().mean() * 100
).round(2)

print("\nMissing values:\n", missing.to_string())
missing.to_csv(OUT_DIR / "missing_values.csv")

# Check whether missing values vary by target class
missing_by_class = []
for col in train.columns:
    if col in [TARGET, ID] or not train[col].isna().any():
        continue
    rates = train.groupby(TARGET, observed=True)[col].apply(
        lambda s: s.isna().mean() * 100
    )
    row = {"feature": col}
    row.update({f"class_{k}_missing_pct": round(v, 2)
                for k, v in rates.items()})
    missing_by_class.append(row)

if missing_by_class:
    missing_class_df = pd.DataFrame(missing_by_class)
    print("\nMissingness by class (%):\n",
          missing_class_df.to_string(index=False))
    missing_class_df.to_csv(
        OUT_DIR / "missingness_by_class.csv", index=False
    )
else:
    print("\nNo missing feature values in training data.")

# Target distribution
plt.figure(figsize=(7, 4))
ax = sns.countplot(data=train, x=TARGET, hue=TARGET, legend=False)
ax.set_title("Training Target Distribution")
ax.set_xlabel("Condition class")
ax.set_ylabel("Number of rows")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(OUT_DIR / "target_distribution.png", dpi=150)
plt.close()

# Numerical feature distributions by class
ncols = 2
nrows = int(np.ceil(len(numeric_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = np.array(axes).reshape(-1)

for ax, col in zip(axes, numeric_cols):
    sns.histplot(
        data=train, x=col, hue=TARGET,
        stat="density", common_norm=False,
        element="step", fill=False, ax=ax
    )
    ax.set_title(f"{col} by condition")

for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(
    OUT_DIR / "feature_distributions_by_class.png", dpi=150
)
plt.close(fig)

# Boxplots by class
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = np.array(axes).reshape(-1)

for ax, col in zip(axes, numeric_cols):
    sns.boxplot(
        data=train, x=TARGET, y=col, hue=TARGET,
        legend=False, ax=ax
    )
    ax.set_title(f"{col} by condition")

for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "feature_boxplots_by_class.png", dpi=150)
plt.close(fig)

# Correlation matrix
corr = train[numeric_cols].corr()
corr.to_csv(OUT_DIR / "correlations.csv")

plt.figure(figsize=(11, 8))
sns.heatmap(
    corr, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, square=True
)
plt.title("Numerical Feature Correlations")
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("\n========== STRONG CORRELATIONS ==========")
pairs = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
).stack().reset_index()
pairs.columns = ["feature_1", "feature_2", "correlation"]
pairs["abs_correlation"] = pairs["correlation"].abs()
print(
    pairs.sort_values("abs_correlation", ascending=False)
    .head(15).round(3).to_string(index=False)
)

# Potential outliers using the IQR rule
print("\n========== POTENTIAL OUTLIERS (IQR RULE) ==========")
outlier_rows = []

for col in numeric_cols:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((train[col] < lower) | (train[col] > upper)).sum()
    outlier_rows.append({
        "feature": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(count),
        "outlier_pct": round(count / len(train) * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_rows)
print(outlier_df.round(3).to_string(index=False))
outlier_df.to_csv(OUT_DIR / "outlier_counts.csv", index=False)

print("\n========== TRAIN / TEST COMPARISON ==========")
comparison_rows = []

for col in numeric_cols:
    a = train[col].dropna()
    b = test[col].dropna()
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    std_diff = (
        (a.mean() - b.mean()) / pooled_std
        if pooled_std > 0 else np.nan
    )
    comparison_rows.append({
        "feature": col,
        "train_mean": a.mean(),
        "test_mean": b.mean(),
        "train_std": a.std(),
        "test_std": b.std(),
        "standardized_mean_difference": std_diff,
    })

comparison = pd.DataFrame(comparison_rows)
print(comparison.round(3).to_string(index=False))
comparison.to_csv(
    OUT_DIR / "train_test_numeric_comparison.csv", index=False
)

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = np.array(axes).reshape(-1)

for ax, col in zip(axes, numeric_cols):
    sns.histplot(
        train[col].dropna(), stat="density",
        element="step", fill=False, label="Train", ax=ax
    )
    sns.histplot(
        test[col].dropna(), stat="density",
        element="step", fill=False, label="Test", ax=ax
    )
    ax.set_title(f"Train vs test: {col}")
    ax.legend()

for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "train_test_distributions.png", dpi=150)
plt.close(fig)

print("\n========== CATEGORICAL FEATURES ==========")
for col in categorical_cols:
    train_categories = set(train[col].dropna().unique())
    test_categories = set(test[col].dropna().unique())

    print(f"\n{col}")
    print("Train categories:", sorted(train_categories, key=str))
    print("Test categories:", sorted(test_categories, key=str))
    print("Only in train:",
          sorted(train_categories - test_categories, key=str))
    print("Only in test:",
          sorted(test_categories - train_categories, key=str))
    print("Train counts:\n",
          train[col].value_counts(dropna=False).to_string())
    print("Test counts:\n",
          test[col].value_counts(dropna=False).to_string())

    counts = pd.concat([
        train[col].value_counts(normalize=True).rename("train"),
        test[col].value_counts(normalize=True).rename("test")
    ], axis=1).fillna(0)

    counts.plot(kind="bar", figsize=(8, 4))
    plt.title(f"Train vs test categories: {col}")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"category_{col}.png", dpi=150)
    plt.close()

print("\n========== PROPOSED FEATURE RELATIONSHIPS ==========")

if {"power_load", "drive_torque", "shaft_speed"}.issubset(train.columns):
    product = train["drive_torque"] * train["shaft_speed"]
    print(
        "Correlation power_load vs torque * speed:",
        round(train["power_load"].corr(product), 4)
    )
    ratio = train["power_load"] / product.replace(0, np.nan)
    print("Power / (torque * speed) summary:\n",
          ratio.describe().round(4).to_string())

if {"thermal_stress", "process_temp", "ambient_temp"}.issubset(train.columns):
    temp_diff = train["process_temp"] - train["ambient_temp"]
    print(
        "Correlation thermal_stress vs temperature difference:",
        round(train["thermal_stress"].corr(temp_diff), 4)
    )

print("\nEDA complete. Results saved to:", OUT_DIR.resolve())
