# =============================================================================
# UCI Heart Disease --- Classification Project
# =============================================================================

print("=" * 60)
print("UCI Heart Disease Data - Classification Project")
print("=" * 60)

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import gaussian_kde
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

print("\n")
print("=" * 60)
print("1. Import Data and Insepct")
print("=" * 60)

# Load in dataset from OpenML
dataset = datasets.fetch_openml(data_id=46590,
                                return_X_y=False,
                                as_frame=True
                                )

# Extract Target and Features
X = dataset.data
y = dataset.target

# Combine into DataFrame
df = X.copy()
df["target"] = y

# Dataset checks
print(f"\n[1.1] Dataset loaded: {dataset.details['name']}")
print(f"\n[1.2] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\n[1.3] First 5 rows of dataset:")
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

# =============================================================================
# 2. EDA & Visualization
# =============================================================================

print("\n")
print("=" * 60)
print("1. EDA & Visualization")
print("=" * 60)

# Target Variable
print("\n[2.1] Target variable (Presence of Disease) Distributiion: ")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot info
target_counts = df["target"].value_counts()
tar_counts = [target_counts[0], target_counts[1]]
tar_labels = ["No Disease", "Disease"]
tar_colors = ["green", "red"]

# Bar plot
tar_bars = axes[0].bar(tar_labels, 
                       tar_counts, 
                       color=tar_colors, 
                       edgecolor="black", 
                       width=0.5)
axes[0].set_title("Heart Disease Distribution (Disease vs No Disease)")
axes[0].set_ylabel("Count")
axes[0].bar_label(tar_bars, padding=1)

# Pie chart
axes[1].pie(tar_counts, 
                      labels=tar_labels, 
                      colors=tar_colors,
                      autopct="%1.1f%%",
                      startangle=90,
                      wedgeprops={"edgecolor": "black"})
axes[1].set_title("Heart Disease Proportion")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_target_distribution.png")

# Define Categorical vs Numerical Feature columns
numerical_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                  "maximum_heart_rate_achieved", "oldpeak",
                  "number_of_major_vessels"]

categorical_cols = ["sex", "chest", "fasting_blood_sugar",
                    "resting_electrocardiographic_results",
                    "exercise_induced_angina", "slope", "thal"]

# Feature Variables

# Plot info
feat_NoDisease_color = "green"
feat_Disease_color = "red"
feat_edge_color = "black"
feat_alpha = 0.5
feat_bins = 20
feat_labels = ["No Disease", "Disease"]

# Bar plots
print("\n[2.2] Numerical feature variables Distributiion (Bar plots): ")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].hist(df[df["target"] == 0][col], bins=feat_bins, alpha=feat_alpha,
                 color=feat_NoDisease_color, label=feat_labels[0], edgecolor=feat_edge_color)
    axes[i].hist(df[df["target"] == 1][col], bins=feat_bins, alpha=feat_alpha,
                 color=feat_Disease_color, label=feat_labels[1], edgecolor=feat_edge_color)
    # KDE lines - NOTE: Used LLM for this. Good to know for future!
    ax2 = axes[i].twinx()
    for target, color in zip([0, 1], [feat_NoDisease_color, feat_Disease_color]):
        vals = df[df["target"] == target][col].dropna()
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 200)
        ax2.plot(x_range, kde(x_range), color=color, linewidth=2)
    ax2.set_ylabel("")
    ax2.set_yticks([])
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend()

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_numerical_distributions.png")

# Boxplots
print("\n[2.3] Numerical feature variables Boxplots: ")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    data_0 = df[df["target"] == 0][col]
    data_1 = df[df["target"] == 1][col]
    bp = axes[i].boxplot([data_0, data_1],
                          patch_artist=True,
                          labels=feat_labels)
    bp["boxes"][0].set_facecolor(feat_NoDisease_color)
    bp["boxes"][1].set_facecolor(feat_Disease_color)
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].set_ylabel("Value")

plt.suptitle("Boxplots of Numerical Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_boxplots.png")

# Categorical feature variables by Target variable
print("\n[2.4] Categorical Feature Counts: ")
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
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(no_disease.index.astype(int))
    axes[i].legend(fontsize=7)

# Hide unused subplot
axes[-1].set_visible(False)

plt.suptitle("Categorical Features by Target Class")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_categorical_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_categorical_counts.png")

# Correlation Plot
print("\n[2.5] Correlation Plot: ")
fig, ax = plt.subplots(figsize=(12, 9))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_correlation.png")

# Key correlations with Target
print("\n[2.6] Key Correlations with target: ")
target_corr = corr["target"].drop("target").sort_values(ascending=False)
print(target_corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["red" if count > 0 else "green" for count in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlations with Target")
ax.set_xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_key_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_key_correlations.png")

# =============================================================================
# 3. Feature Engineering
# =============================================================================

print("\n")
print("=" * 60)
print("3. Feature Engineering")
print("=" * 60)

df_engin = df.copy()

# Remove weakest correlated variable to target
print("\n[3.1] Drop weakeast correlated feature to target: fasting_blood_sugar (r = -0.016)")
df_engin = df_engin.drop(columns=["fasting_blood_sugar"])

# Final Feature List
X = df_engin.drop(columns=["target"])
y = df_engin["target"]

print(f"\n[3.2] Final Feature List: ")
for col in X.columns:
    print(f"    - {col}")

# Perform Train/Test split

print("\n[3.3] Split raw data 80/20 for Random Tree and XGBoost Models :")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"  X_train_raw    shape : {X_train_raw.shape} → for RF, XGBoost")
print(f"  X_test_raw shape : {X_test_raw.shape} → for RF, XGBoost")

print("\n[3.4] Split scaled data 80/20 for Logistic Regression and SVM Models :")
scaler = StandardScaler()

X_train_scaled = X_train_raw.copy()
X_test_scaled = X_test_raw.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_raw[numerical_cols])
X_test_scaled[numerical_cols]  = scaler.transform(X_test_raw[numerical_cols])

print(f"  X_train_scaled    shape : {X_train_raw.shape} → for LR, SVM")
print(f"  X_test_scaled shape : {X_test_scaled.shape} → for LR, SVM")

# =============================================================================
# 4. Modeling and Performance Metrics
# =============================================================================

print("\n")
print("=" * 60)
print("5. Modeling and Performance Metrics")
print("=" * 60)

results = {}

# Logistic Regression

log_reg_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)

log_reg_pred = log_reg_model.predict(X_test_scaled)
log_reg_prob = log_reg_model.predict_proba(X_test_scaled)[:, 1]

results["Logistic Regression"] = {
    "AUC-ROC": roc_auc_score(y_test, log_reg_prob),
    "Accuracy": accuracy_score(y_test, log_reg_pred),
    "F1 Score": f1_score(y_test, log_reg_pred)
}

print("\n[4.1] Logistic Regression Model..... Complete ✓")

# Support Vector Machine (SVM)

svm_model = SVC(random_state=RANDOM_STATE, probability=True)
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

results["SVM"] = {
    "AUC-ROC": roc_auc_score(y_test, svm_prob),
    "Accuracy": accuracy_score(y_test, svm_pred),
    "F1 Score": f1_score(y_test, svm_pred)
}

print("\n[4.2] SVM Model..... Complete ✓")

# Random Forest

rand_for_grid = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_features": ["sqrt", "log2", "None"],
    "max_depth": [3, 5, 10, None]
}

rand_for_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE), 
    param_grid=rand_for_grid, 
    scoring="roc_auc",
    cv=5
)

rand_for_grid.fit(X_train_raw, y_train)

rand_for_model = rand_for_grid.best_estimator_

rand_for_pred = rand_for_model.predict(X_test_raw)
rand_for_prob = rand_for_model.predict_proba(X_test_raw)[:, 1]

results["Random Forest"] = {
    "AUC-ROC": roc_auc_score(y_test, rand_for_prob),
    "Accuracy": accuracy_score(y_test, rand_for_pred),
    "F1 Score": f1_score(y_test, rand_for_pred)
}

print("\n[4.3] Random Forest Model..... Complete ✓")

# XGBoost

xgb_grid = {
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
    "max_depth": [3, 5, 7]
}

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE), 
    param_grid=xgb_grid, 
    scoring="roc_auc",
    cv=5
)

xgb_grid.fit(X_train_raw, y_train)

xgb_model = xgb_grid.best_estimator_

xgb_pred = xgb_model.predict(X_test_raw)
xgb_prob = xgb_model.predict_proba(X_test_raw)[:, 1]


results["XGBoost"] = {
    "AUC-ROC": roc_auc_score(y_test, xgb_prob),
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "F1 Score": f1_score(y_test, xgb_pred)
}

print("\n[4.4] XGBoost Model..... Complete ✓")

print("\n[4.5] Performance Metrics: ")

results_df = pd.DataFrame(results).T
print(results_df.to_string())

print("\n[4.6] Model Performance Comparison Plot: ")

fig, axes = plt.subplots(1, 3, figsize=(15,5))

perf_metrics = ["AUC-ROC", "Accuracy", "F1 Score"]
perf_colors = ["green", "blue", "purple", "red"]

for i, perf_metrics in enumerate(perf_metrics):
    values = results_df[perf_metrics]
    bars = axes[i].bar(results_df.index, values, color=perf_colors, edgecolor="black")
    axes[i].set_title(perf_metrics, fontsize=12)
    axes[i].set_ylabel("Score")
    axes[i].set_xticklabels(results_df.index)
    for bar, count in zip(bars, values):
        axes[i].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        round(count, 3),
        ha="center"
    )

plt.suptitle("Model Performance Comparison")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_model_comparison.png")
