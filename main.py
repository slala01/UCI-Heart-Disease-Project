# =============================================================================
# UCI Heart Disease --- Classification Project
# =============================================================================

print("=" * 60)
print("UCI Heart Disease Data - Classification Project")
print("=" * 60)

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from scipy.stats import gaussian_kde
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# File Configurations
RANDOM_STATE = 67
OUTPUT_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# 1. Import Data and Insepct
# =============================================================================

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
print("-" * 60)
print(df.head())
print("\n[1.4] Descriptive Statistics:")
print("-" * 60)
print(df.describe())
print(f"\n[1.5] Missing Values:")
print("-" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
if missing_df.empty: 
    print("No missing values!")
else:
    print(missing_df)

# Recode target variable to binary
df["target"] = df["target"].apply(lambda x: 1 if x ==  "present" else 0)
print("\n")

# =============================================================================
# 2. EDA & Visualization
# =============================================================================

print("=" * 60)
print("2. EDA & Visualization")
print("=" * 60)

# Target Variable
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
print("\n[2.1] Saving Target Distribution plot as 01_target_distribution.png..... Complete ✓")

# Define Categorical vs Numerical Feature columns
numerical_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                  "maximum_heart_rate_achieved", "oldpeak",
                  "number_of_major_vessels"]

categorical_cols = ["sex", "chest", "fasting_blood_sugar",
                    "resting_electrocardiographic_results",
                    "exercise_induced_angina", "slope", "thal"]

# Feature Variables

# Bar plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].hist(df[df["target"] == 0][col], bins=20, alpha=0.5,
                 color="green", label="No Disease", edgecolor="black")
    axes[i].hist(df[df["target"] == 1][col], bins=20, alpha=0.5,
                 color="red", label="Disease", edgecolor="black")
    # KDE lines
    ax2 = axes[i].twinx()
    for target, color in zip([0, 1], ["green", "red"]):
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
print("\n[2.2] Saving Numerical Features Distribution plot as 02_numerical_distributions.png..... Complete ✓")

# Categorical feature variables by Target variable
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    no_disease = df[df["target"] == 0][col].value_counts().sort_index()
    disease = df[df["target"] == 1][col].value_counts().sort_index()
    x = np.arange(len(no_disease.index))
    width = 0.35
    axes[i].bar(x - width/2, no_disease.values, width,
                label="No Disease", color="green", edgecolor="black")
    axes[i].bar(x + width/2, disease.reindex(no_disease.index).fillna(0).values,
                width, label="Disease", color="red", edgecolor="black")
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(no_disease.index.astype(int))
    axes[i].legend(fontsize=7)

# Hide unused subplot
axes[-1].set_visible(False)

plt.suptitle("Categorical Features by Target Class")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_categorical_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[2.3] Saving Categotrical Counts plot as 03_categorical_counts.png..... Complete ✓")

# Correlation Plot
fig, ax = plt.subplots(figsize=(12, 9))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_correlation_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[2.4] Saving Correlation Plot as 04_correlation_plot.png..... Complete ✓")

# Key correlations with Target
print("\n[2.5] Key Correlations with target: ")
print("-" * 60)
target_corr = corr["target"].drop("target").sort_values(ascending=False)
print(target_corr.round(3))
print("\n")

# =============================================================================
# 3. Feature Engineering
# =============================================================================

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
print("-" * 60)
for col in X.columns:
    print(f"    - {col}")

# Perform Train/Test split
print("\n[3.3] Train/Test split raw data 75/25 for Random Forest and XGBoost Models: ")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"\n X_train_raw shape : {X_train_raw.shape} → for Random Forest and XGBoost models")
print(f" X_test_raw  shape : {X_test_raw.shape}  → for Random Forest and XGBoost models")

print("\n[3.4] Transform skewed variables and scale data for Logistic Regression and SVM Models:")
scaler = StandardScaler()

X_train_scaled = X_train_raw.copy()
X_test_scaled  = X_test_raw.copy()

# Log transform skewed features (LR and SVM only)
skewed_cols = ["serum_cholestoral", "oldpeak"]

X_train_scaled[skewed_cols] = np.log1p(X_train_scaled[skewed_cols])
X_test_scaled[skewed_cols]  = np.log1p(X_test_scaled[skewed_cols])

print(f"\n Transforming serum_cholestoral ..... Complete ✓")
print(f" Transforming oldpeak ..... Complete ✓")

# Scale data
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_scaled[numerical_cols])
X_test_scaled[numerical_cols]  = scaler.transform(X_test_scaled[numerical_cols])

print("\n[3.5] Train/Test split scaled data 75/25 for Logisitic Regression and SVM Models: ")
print(f"\n X_train_scaled shape : {X_train_scaled.shape} → for Logistic Regression and SVM models")
print(f" X_test_scaled  shape : {X_test_scaled.shape}  → for Logistic Regression and SVM models")
print("\n")

# =============================================================================
# 4. Modeling and Performance Metrics
# =============================================================================

print("=" * 60)
print("4. Modeling and Performance Metrics")
print("=" * 60)

results = {}

# Logistic Regression
log_reg_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

log_reg_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 
    param_grid=log_reg_grid, 
    scoring="roc_auc",
    cv=5
)

log_reg_grid.fit(X_train_scaled, y_train)

log_reg_model = log_reg_grid.best_estimator_

log_reg_pred = log_reg_model.predict(X_test_scaled)
log_reg_prob = log_reg_model.predict_proba(X_test_scaled)[:, 1]

results["Logistic Regression"] = {
    "AUC-ROC": roc_auc_score(y_test, log_reg_prob),
    "Accuracy": accuracy_score(y_test, log_reg_pred),
    "Recall": round(recall_score(y_test, log_reg_pred), 4),
    "F1 Score": f1_score(y_test, log_reg_pred)
}

print("\n[4.1] Logistic Regression Model..... Complete ✓")

# Support Vector Machine (SVM)
svm_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"]
}

svm_grid = GridSearchCV(
    SVC(random_state=RANDOM_STATE, probability=True), 
    param_grid=svm_grid, 
    scoring="roc_auc",
    cv=5
)

svm_grid.fit(X_train_scaled, y_train)

svm_model = svm_grid.best_estimator_

svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

results["SVM"] = {
    "AUC-ROC": roc_auc_score(y_test, svm_prob),
    "Accuracy": accuracy_score(y_test, svm_pred),
    "Recall": round(recall_score(y_test, svm_pred), 4),
    "F1 Score": f1_score(y_test, svm_pred)
}

print("\n[4.2] SVM Model..... Complete ✓")

# Random Forest
rand_for_grid = {
    "n_estimators": [100, 300, 500],
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
    "Recall": round(recall_score(y_test, rand_for_pred), 4),
    "F1 Score": f1_score(y_test, rand_for_pred)
}

print("\n[4.3] Random Forest Model..... Complete ✓")

# XGBoost
xgb_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
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
    "Recall": round(recall_score(y_test, xgb_pred), 4),
    "F1 Score": f1_score(y_test, xgb_pred)
}

print("\n[4.4] XGBoost Model..... Complete ✓")

# Performance Metrics
print("\n[4.5] Performance Metrics: ")

results_df = pd.DataFrame(results).T
print("-" * 60)
print(results_df.to_string())

fig, axes = plt.subplots(2, 2, figsize=(15,9))
axes = axes.flatten()

perf_metrics = ["AUC-ROC", "Accuracy", "Recall", "F1 Score"]
model_colors = ["green", "blue", "purple", "red"]

for i, perf_metrics in enumerate(perf_metrics):
    values = results_df[perf_metrics]
    bars = axes[i].bar(results_df.index, values, color=model_colors, edgecolor="black")
    axes[i].set_title(perf_metrics, fontsize=12)
    axes[i].set_ylabel("Score")
    axes[i].set_ylim(0, 1.10)
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
plt.savefig(f"{PLOTS_DIR}/05_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4.6] Saving Model Comparison plot as 05_model_comparison.png..... Complete ✓")

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(log_reg_model,  X_test_scaled, y_test, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(svm_model,  X_test_scaled, y_test, ax=ax, name="SVM")
RocCurveDisplay.from_estimator(rand_for_model, X_test_raw, y_test, ax=ax, name="Random Forest")
RocCurveDisplay.from_estimator(xgb_model, X_test_raw, y_test, ax=ax, name="XGBoost")

ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
ax.set_title("ROC Curves - All Models", fontweight="bold", fontsize=12)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4.7] Saving ROC Curves plot as 06_roc_curves.png..... Complete ✓")

# Confusion Matrices
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

confus_df = [
    ("Logistic Regression", log_reg_model, X_test_scaled, log_reg_pred),
    ("SVM", svm_model, X_test_scaled, svm_pred),
    ("Random Forest", rand_for_model, X_test_raw, rand_for_pred),
    ("XGBoost", xgb_model, X_test_raw, xgb_pred)
]

for i, (name, model, X_test_m, Y_pred) in enumerate(confus_df):
    cm   = confusion_matrix(y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Disease", "Disease"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(name, fontweight="bold", fontsize=10)

plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4.8] Saving Confusion Matrices as 07_confusion_matrices.png..... Complete ✓")
print("\n")

# =============================================================================
# 5. Model Diagnostic
# =============================================================================

print("=" * 60)
print("5. Model Diagnostic")
print("=" * 60)

# Train vs Test AUC Gaps
print("\n[5.1] Train vs Test AUC Gap: ")

model_registry = {
    "Logistic Regression": (log_reg_model,  X_train_scaled, X_test_scaled),
    "SVM": (svm_model,  X_train_scaled,    X_test_scaled),
    "Random Forest": (rand_for_model, X_train_raw,    X_test_raw),
    "XGBoost": (xgb_model, X_train_raw, X_test_raw)
}

diagnostics = {}

for name, (model, X_train_m, X_test_m) in model_registry.items():
    train_auc = roc_auc_score(y_train,
                               model.predict_proba(X_train_m)[:, 1])
    test_auc  = roc_auc_score(y_test,
                               model.predict_proba(X_test_m)[:, 1])
    gap       = train_auc - test_auc

    diagnostics[name] = {
        "Train AUC"      : round(train_auc, 4),
        "Test AUC"       : round(test_auc, 4),
        "Gap"            : round(gap, 4),
    }

print("-" * 60)
diag_df = pd.DataFrame(diagnostics).T
print(diag_df.to_string())

# Threshold adjustments
print("\n[5.2] Adjusted Threshold Check: ")
print("-" * 60)

threshold_models = {
    "Logistic Regression": (log_reg_model,  X_test_scaled),
    "SVM": (svm_model, X_test_scaled),
    "Random Forest": (rand_for_model,  X_test_raw),
    "XGBoost": (xgb_model, X_test_raw)
}

thresholds = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]

for name, (model, X_test_m) in threshold_models.items():
    print(f"\n  {name}:")
    print(f"  {'-'*60}")
    print(f"  {'Threshold':<12} {'Recall':<10} {'AUC-ROC':<12} {'F1':<10} {'Accuracy':<10}")
    print(f"  {'-'*60}")
    probs = model.predict_proba(X_test_m)[:, 1]
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        recall = recall_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        print(f"  {thresh:<12} {recall:<10.3f} {auc:<12.3f} {f1:<10.3f} {accuracy:<10.3f}")

print("\n[5.3] Optimal Model Threshold: Random Forest @ 0.44 Threshold")

rand_for_pred_44 = (rand_for_prob >= 0.44).astype(int)

# ROC Curve for Random Forest at 0.44 Threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, thresholds_roc = roc_curve(y_test, rand_for_prob)
auc_score = roc_auc_score(y_test, rand_for_prob)

axes[0].plot(fpr, tpr, color="red", lw=2,
              label=f"Random Forest (AUC = {auc_score:.3f})")
axes[0].scatter(fpr[np.argmin(np.abs(thresholds_roc - 0.44))],
                 tpr[np.argmin(np.abs(thresholds_roc - 0.44))],
                 color="black", zorder=5, s=100,
                 label="Threshold = 0.44")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
axes[0].set_title("ROC Curve: Random Forest @ 0.44 Threshold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Confusion Matrix for Random Forest at 0.44 Threshold
cm   = confusion_matrix(y_test, rand_for_pred_44)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Disease", "Disease"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix: Random Forest @ 0.44 Threshold")

# Performance Metrics for Random Forest at 0.44 Threshold
recall   = recall_score(y_test, rand_for_pred_44)
accuracy = accuracy_score(y_test, rand_for_pred_44)
f1       = f1_score(y_test, rand_for_pred_44)

axes[1].set_xlabel(
    f"Recall: {recall:.3f}  |  Accuracy: {accuracy:.3f}  |  F1: {f1:.3f}"
)

plt.suptitle("Random Forest: Optimal Threshold @ 0.44")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_random_forest_optimal_threshold.png",
             dpi=150, bbox_inches="tight")
plt.close()
print("Saving ROC Curve and Confusion Matrix plot as 08_random_forest_optimal_threshold.png..... Complete ✓")