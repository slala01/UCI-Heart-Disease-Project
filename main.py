# =============================================================================
# UCI Heart Disease --- Classification Project
# =============================================================================

# Imports #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# Config #
RANDOM_STATE = 67
OUTPUT_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# 1. Import Data and Insepct
# =============================================================================

# Load in dataset from OpenML
dataset = datasets.fetch_openml(data_id=46590,
                                return_X_y=False,
                                as_frame=True
                                )

# Extract Target and Features
X = dataset.data
Y = dataset.target

# Combine into DataFrame
df = X.copy()
df["target"] = Y

# Dataset checks
print(f"\n[1.1] Dataset loaded: {dataset.details['name']}")
print(f"\n[1.2] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\n[1.3] First 5 rows:")
print(df.head())
print("\n[1.4] Descriptive Statistics:")
print(df.describe())
print(f"\n[1.5] Missing Values:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
if missing_df.empty: 
    print("No missing values!")
else:
    print(missing_df)

# Recode target variable to binary
df["target"] = df["target"].apply(lambda x: 1 if x ==  "present" else 0)
print(f"\n[1.7] No Disease: {(df['target'] == 0).sum()}; Disease: {(df['target'] == 1).sum()}")

# =============================================================================
# 2. EDA & Visualization
# =============================================================================

# Define Categorical vs numerical columns
categorical_cols = ["sex", "chest", "fasting_blood_sugar",
                    "resting_electrocardiographic_results",
                    "exercise_induced_angina", "slope", "thal"]

numerical_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                  "maximum_heart_rate_achieved", "oldpeak",
                  "number_of_major_vessels"]

# Target Variable
print("\n[2.1] Target variable (Presence of Disease) Distributiion: ")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count plot
target_counts = df["target"].value_counts()
axes[0].bar(["No Disease", "Disease"],
            [target_counts[0], target_counts[1]],
            color=["green", "red"], edgecolor="black", width=0.5)
axes[0].set_title("Target Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
for i, count in enumerate([target_counts[0], target_counts[1]]):
    axes[0].text(i, count + 1, str(count), ha="center", fontweight="bold")

# Pie chart
axes[1].pie([target_counts[0], target_counts[1]],
            labels=["No Disease", "Disease"],
            colors=["green", "red"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "black"})
axes[1].set_title("Target Class Proportion", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_target_distribution.png")

# Feature Variables
print("\n[2.2] Numerical feature variables Distributiion: ")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[df["target"] == 0][col], bins=20, alpha=0.6,
                 color="green", label="No Disease", edgecolor="black")
    axes[i].hist(df[df["target"] == 1][col], bins=20, alpha=0.6,
                 color="red", label="Disease", edgecolor="black")
    axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend()

plt.suptitle("Numerical Feature Distributions by Target", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_numerical_distributions.png")

# Numerical feature variables vs Target variable Boxplots
print("\n[2.3] Numerical feature variables vs Target variable Boxplot: ")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    data_0 = df[df["target"] == 0][col]
    data_1 = df[df["target"] == 1][col]
    bp = axes[i].boxplot([data_0, data_1],
                          patch_artist=True,
                          labels=["No Disease", "Disease"])
    bp["boxes"][0].set_facecolor("green")
    bp["boxes"][1].set_facecolor("red")
    axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
    axes[i].set_ylabel("Value")

plt.suptitle("Numerical Features by Target Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_boxplots.png")

# Categorical feature variables by Target variable
print("\n[2.4] Plotting categorical feature counts...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    no_disease = df[df["target"] == 0][col].value_counts().sort_index()
    disease = df[df["target"] == 1][col].value_counts().sort_index()
    x = np.arange(len(no_disease.index))
    width = 0.35
    axes[i].bar(x - width/2, no_disease.values, width,
                label="No Disease", color="#2ecc71", edgecolor="black")
    axes[i].bar(x + width/2, disease.reindex(no_disease.index).fillna(0).values,
                width, label="Disease", color="#e74c3c", edgecolor="black")
    axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(no_disease.index.astype(int))
    axes[i].legend(fontsize=7)

# Hide unused subplot
axes[-1].set_visible(False)

plt.suptitle("Categorical Features by Target Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_categorical_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_categorical_counts.png")

# Correlation Plot
print("\n[2.5] Plotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(12, 9))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_correlation.png")

# Key correlations with Target
print("\n[2.6] Feature correlations with target:")
target_corr = df.corr()["target"].drop("target").sort_values(ascending=False)
print(target_corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["red" if v > 0 else "green" for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlations with Target", fontweight="bold")
ax.set_xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_target_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_target_correlations.png")