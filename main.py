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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
from xgboost import XGBClassifier
import logging
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# Logging Configurations
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# File Configurations
RANDOM_STATE = 67
OUTPUT_DIR   = "outputs"
PLOTS_DIR    = "outputs/plots"
MODELS_DIR   = "outputs/models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# UCI Heart Disease --- Classification Project
# =============================================================================

def save_plot(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saving: %s..... Complete ✓", filename)


def save_model(model, model_name):
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    logger.info("  Model saved → %s", path)


def evaluate_model(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "AUC-ROC"   : round(roc_auc_score(y_test, prob), 4),
        "Accuracy"  : round(accuracy_score(y_test, pred), 4),
        "Recall"    : round(recall_score(y_test, pred), 4),
        "F1 Score"  : round(f1_score(y_test, pred), 4),
        "Precision" : round(precision_score(y_test, pred), 4)}

def main():

    # =============================================================================
    # 1. Import Data and Inspect
    # =============================================================================

    print( )
    print("=" * 60)
    print("1. Import Data and Inspect")
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
    logger.info("[1.1] Dataset loaded: %s", dataset.details["name"])
    logger.info("[1.2] Dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("[1.3] First 5 rows of dataset:")
    logger.info("\n%s", df.head().to_string())
    logger.info("[1.4] Descriptive Statistics:")
    logger.info("\n%s", df.describe().to_string())
    logger.info("[1.5] Missing Values:")

    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100)
    missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    if missing_df["Missing Count"].sum() == 0:
        logger.info("  No missing values!")
    else:
        logger.warning("  Missing values detected:\n%s", missing_df[missing_df["Missing Count"] > 0].to_string())

    # Recode target variable to binary
    df["target"] = df["target"].apply(lambda x: 1 if x ==  "present" else 0)
    logger.info("[1.6] Target variable encoded to binary")

    # =============================================================================
    # 2. EDA & Visualization
    # =============================================================================

    print( )
    print("=" * 60)
    print("2. EDA & Visualization")
    print("=" * 60)

    # Target Variable
    def target_distribution_plot(df):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        counts = [(df["target"] == 0).sum(), (df["target"] == 1).sum()]
        labels = ["No Disease", "Disease"]
        colors = ["green", "red"]

        bars = axes[0].bar(labels, counts, color=colors, edgecolor="black", width=0.5)
        axes[0].set_title("Heart Disease Distribution (Disease vs No Disease)")
        axes[0].set_ylabel("Count")
        axes[0].bar_label(bars, padding=1)

        axes[1].pie(counts, labels=labels, colors=colors,
                        autopct="%1.1f%%", startangle=90,
                        wedgeprops={"edgecolor": "black"})
        axes[1].set_title("Heart Disease Proportion")
        plt.tight_layout()

        return fig

    # Define Categorical vs Numerical Feature columns
    numerical_cols = ["age", "resting_blood_pressure", "serum_cholestoral",
                        "maximum_heart_rate_achieved", "oldpeak",
                        "number_of_major_vessels"]

    categorical_cols = ["sex", "chest", "fasting_blood_sugar",
                            "resting_electrocardiographic_results",
                            "exercise_induced_angina", "slope", "thal"]
    
    # Numerical Features
    def numerical_distribution_plot(df):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, col in enumerate(NUMERICAL_COLS):
            axes[i].hist(df[df["target"] == 0][col], bins=20, alpha=0.5,
                        color="green", label="No Disease", edgecolor="black")
            axes[i].hist(df[df["target"] == 1][col], bins=20, alpha=0.5,
                        color="red", label="Disease", edgecolor="black")
            ax2 = axes[i].twinx()
            for target, color in zip([0, 1], ["green", "red"]):
                vals    = df[df["target"] == target][col].dropna()
                kde     = gaussian_kde(vals)
                x_range = np.linspace(vals.min(), vals.max(), 200)
                ax2.plot(x_range, kde(x_range), color=color, linewidth=2)
            ax2.set_yticks([])
            axes[i].set_title(col.replace("_", " ").title())
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")
            axes[i].legend()
        plt.suptitle("Distribution of Numerical Features")
        plt.tight_layout()

        return fig
    
    # Categorical Features
    def categorical_distribution_plot(df):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        for i, col in enumerate(CATEGORICAL_COLS):
            no_disease = df[df["target"] == 0][col].value_counts().sort_index()
            disease    = df[df["target"] == 1][col].value_counts().sort_index()
            x          = np.arange(len(no_disease.index))
            width      = 0.35

            axes[i].bar(x - width / 2, no_disease.values, width,
                        label="No Disease", color="green", edgecolor="black")
            axes[i].bar(x + width / 2,
                        disease.reindex(no_disease.index).fillna(0).values,
                        width, label="Disease", color="red", edgecolor="black")
            axes[i].set_title(col.replace("_", " ").title())
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(no_disease.index)
            axes[i].legend(fontsize=7)
        axes[-1].set_visible(False)
        plt.suptitle("Categorical Features by Target Class")
        plt.tight_layout()

        return fig
    
    # Correlation Plot
    def correlation_plot(df):
        fig, ax = plt.subplots(figsize=(12, 9))

        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title("Feature Correlation")
        plt.tight_layout()

        return fig

    eda_plots = [
        ("01_target_distribution.png", target_distribution_plot),
        ("02_numerical_distributions.png", numerical_distribution_plot),
        ("03_categorical_counts.png", categorical_distribution_plot),
        ("04_correlation_plot.png", correlation_plot)]

    for filename, plot_fn in tqdm(eda_plots, desc="EDA plots", unit="plot"):
        fig = plot_fn(df)
        save_plot(fig, filename)

    # =============================================================================
    # 3. Feature Engineering
    # =============================================================================

    print( )
    print("=" * 60)
    print("3. Feature Engineering")
    print("=" * 60)

    df_engin = df.copy()

    # One Hot Encoding for categorical features
    logger.info("[3.1] One Hot Encoding categorical features...")
    df_engin = pd.get_dummies(df_engin, columns=categorical_cols, dtype=int, drop_first=False)
    logger.info("  Encoded columns      : %s", categorical_cols)
    logger.info("  Shape after encoding : %s", df_engin.shape)

    # Dataset check
    logger.info("[3.2] Updated Descriptive Statistics:")
    logger.info("\n%s", df_engin.describe().to_string())

    # Final Feature List
    X = df_engin.drop(columns=["target"])
    y = df_engin["target"]
    feature_names = X.columns.tolist()
    logger.info("[3.3] Total features: %d", len(feature_names))

    # Train/Test split
    logger.info("[3.4] Train/Test split 75/25 (stratified):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    logger.info("  X_train shape : %s", X_train.shape)
    logger.info("  X_test  shape : %s", X_test.shape)

    # =============================================================================
    # 4. Modeling and Performance Metrics
    # =============================================================================

    print( )
    print("=" * 60)
    print("4. Modeling and Performance Metrics")
    print("=" * 60)

    # Model Registry
    def build_model_registry():
        log_reg_pipeline = Pipeline([("scaler", StandardScaler()),
                                     ("model",  LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))])

        svm_pipeline = Pipeline([("scaler", StandardScaler()),
                                 ("model",  SVC(random_state=RANDOM_STATE, probability=True))])

        return {"Logistic Regression": (log_reg_pipeline, {"model__C": [0.01, 0.1, 1, 10, 100],
                                                           "model__penalty": ["l1", "l2"],
                                                           "model__solver": ["liblinear", "saga"]}),

                "SVM": (svm_pipeline, {"model__C": [0.1, 1, 10, 100, 1000],
                                       "model__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                                       "model__kernel": ["rbf"]}),
            
                "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), {"n_estimators": [100, 300, 500],
                                                                                      "max_features": ["sqrt", "log2"],
                                                                                      "max_depth": [3, 5, 7],
                                                                                      "min_samples_split": [5, 10, 20],
                                                                                      "min_samples_leaf": [3, 5, 10]}),
                
                "XGBoost": (XGBClassifier(random_state=RANDOM_STATE), {"n_estimators": [50, 100, 200],
                                                                       "learning_rate": [0.01, 0.1],
                                                                       "max_depth": [2, 3, 4],
                                                                       "subsample": [0.6, 0.8],
                                                                       "colsample_bytree": [0.6, 0.8],
                                                                       "reg_alpha": [0, 0.1],
                                                                       "reg_lambda": [1.0, 5.0]})}























# =============================================================================
# 4. Modeling and Performance Metrics
# =============================================================================

print("=" * 60)
print("4. Modeling and Performance Metrics")
print("=" * 60)

def save_model(model, model_name):
    """Save trained model to pickle file."""
    filename = model_name.lower().replace(" ", "_")
    filepath = f"{MODELS_DIR}/{filename}.pkl"
    joblib.dump(model, filepath)
    print(f"  Model saved → {filepath}")

results = {}

# Logistic Regression
log_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
])

log_reg_param = {
    "model__C"          : [0.01, 0.1, 1, 10, 100],
    "model__penalty"    : ["l1", "l2"],
    "model__solver"     : ["liblinear", "saga"]
}

log_reg_grid = GridSearchCV(
    log_reg_pipeline,
    param_grid=log_reg_param,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

log_reg_grid.fit(X_train_raw, y_train)

log_reg_model = log_reg_grid.best_estimator_

save_model(log_reg_model, "Logistic Regression")

log_reg_pred = log_reg_model.predict(X_test_raw)
log_reg_prob = log_reg_model.predict_proba(X_test_raw)[:, 1]

results["Logistic Regression"] = {
    "AUC-ROC"   : round(roc_auc_score(y_test, log_reg_prob), 4),
    "Accuracy"  : round(accuracy_score(y_test, log_reg_pred), 4),
    "Recall"    : round(recall_score(y_test, log_reg_pred), 4),
    "F1 Score"  : round(f1_score(y_test, log_reg_pred), 4),
    "Precision" : round(precision_score(y_test, log_reg_pred), 4)
}

print("\n[4.1] Logistic Regression Model..... Complete ✓")

# Support Vector Machine (SVM)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  SVC(random_state=RANDOM_STATE, probability=True))
])

svm_param = {
    "model__C"      : [0.1, 1, 10, 100, 1000],
    "model__gamma"  : [1, 0.1, 0.01, 0.001, 0.0001],
    "model__kernel" : ["rbf"]
}

svm_grid = GridSearchCV(
    svm_pipeline, 
    param_grid=svm_param, 
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

svm_grid.fit(X_train_raw, y_train)

svm_model = svm_grid.best_estimator_

save_model(svm_model, "SVM")

svm_pred = svm_model.predict(X_test_raw)
svm_prob = svm_model.predict_proba(X_test_raw)[:, 1]

results["SVM"] = {
    "AUC-ROC"   : round(roc_auc_score(y_test, svm_prob), 4),
    "Accuracy"  : round(accuracy_score(y_test, svm_pred), 4),
    "Recall"    : round(recall_score(y_test, svm_pred), 4),
    "F1 Score"  : round(f1_score(y_test, svm_pred), 4),
    "Precision" : round(precision_score(y_test, svm_pred), 4)
}

print("\n[4.2] SVM Model..... Complete ✓")

# Random Forest
rand_for_param = {
    "n_estimators"      : [100, 300, 500],
    "max_features"      : ["sqrt", "log2"],
    "max_depth"         : [3, 5, 7],
    "min_samples_split" : [5, 10, 20],
    "min_samples_leaf"  : [3, 5, 10]
}

rand_for_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE), 
    param_grid=rand_for_param, 
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

rand_for_grid.fit(X_train_raw, y_train)

rand_for_model = rand_for_grid.best_estimator_

save_model(rand_for_model, "Random Forest") 

rand_for_pred = rand_for_model.predict(X_test_raw)
rand_for_prob = rand_for_model.predict_proba(X_test_raw)[:, 1]

results["Random Forest"] = {
    "AUC-ROC"   : round(roc_auc_score(y_test, rand_for_prob), 4),
    "Accuracy"  : round(accuracy_score(y_test, rand_for_pred), 4),
    "Recall"    : round(recall_score(y_test, rand_for_pred), 4),
    "F1 Score"  : round(f1_score(y_test, rand_for_pred), 4),
    "Precision" : round(precision_score(y_test, rand_for_pred), 4)
}

print("\n[4.3] Random Forest Model..... Complete ✓")

# XGBoost
xgb_param = {
    "n_estimators"      : [50, 100, 200],    
    "learning_rate"     : [0.01, 0.1],
    "max_depth"         : [2, 3, 4],
    "subsample"         : [0.6, 0.8],
    "colsample_bytree"  : [0.6, 0.8],
    "reg_alpha"         : [0, 0.1],
    "reg_lambda"        : [1.0, 5.0]
}

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE), 
    param_grid=xgb_param, 
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

xgb_grid.fit(X_train_raw, y_train)

xgb_model = xgb_grid.best_estimator_

save_model(xgb_model, "XGBoost")

xgb_pred = xgb_model.predict(X_test_raw)
xgb_prob = xgb_model.predict_proba(X_test_raw)[:, 1]


results["XGBoost"] = {
    "AUC-ROC"   : round(roc_auc_score(y_test, xgb_prob), 4),
    "Accuracy"  : round(accuracy_score(y_test, xgb_pred), 4),
    "Recall"    : round(recall_score(y_test, xgb_pred), 4),
    "F1 Score"  : round(f1_score(y_test, xgb_pred), 4),
    "Precision" : round(precision_score(y_test, xgb_pred), 4)
}

print("\n[4.4] XGBoost Model..... Complete ✓")

# Performance Metrics
print("\n[4.5] Performance Metrics: ")

results_df = pd.DataFrame(results).T
print("-" * 60)
print(results_df.to_string())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

perf_metrics = ["AUC-ROC", "Accuracy", "Recall", "F1 Score", "Precision"]
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
        
# Hide the unused 6th subplot
axes[5].set_visible(False)

plt.suptitle("Model Performance Comparison")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4.6] Saving Model Comparison plot as 05_model_comparison.png..... Complete ✓")

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(log_reg_model,  X_test_raw, y_test, ax=ax, color="green", name="Logistic Regression")
RocCurveDisplay.from_estimator(svm_model,  X_test_raw, y_test, ax=ax, color="blue", name="SVM")
RocCurveDisplay.from_estimator(rand_for_model, X_test_raw, y_test, ax=ax, color="purple", name="Random Forest")
RocCurveDisplay.from_estimator(xgb_model, X_test_raw, y_test, ax=ax, color="red", name="XGBoost")

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
    ("Logistic Regression", log_reg_model, X_test_raw, log_reg_pred),
    ("SVM", svm_model, X_test_raw, svm_pred),
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
    "Logistic Regression": (log_reg_model,  X_train_raw, X_test_raw),
    "SVM": (svm_model,  X_train_raw,    X_test_raw),
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
    "Logistic Regression": (log_reg_model,  X_test_raw),
    "SVM": (svm_model, X_test_raw),
    "Random Forest": (rand_for_model,  X_test_raw),
    "XGBoost": (xgb_model, X_test_raw)
}

thresholds = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]

for name, (model, X_test_m) in threshold_models.items():
    print(f"\n  {name}:")
    print(f"  {'-'*60}")
    print(f"  {'Threshold':<12} {'Recall':<10} {'AUC-ROC':<12} {'F1':<10} {'Accuracy':<10} {'Precision':<10}")
    print(f"  {'-'*60}")
    probs = model.predict_proba(X_test_m)[:, 1]
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        recall = recall_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        print(f"  {thresh:<12} {recall:<10.3f} {auc:<12.3f} {f1:<10.3f} {accuracy:<10.3f} {prec:<10.3f}")

print("\n[5.3] Optimal Model Threshold: Random Forest @ 0.45 Threshold")

rand_for_pred_45 = (rand_for_prob >= 0.45).astype(int)

# ROC Curve for Random Forest at 0.45 Threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, thresholds_roc = roc_curve(y_test, rand_for_prob)
auc_score = roc_auc_score(y_test, rand_for_prob)

axes[0].plot(fpr, tpr, color="purple", lw=2,
              label=f"Random Forest (AUC = {auc_score:.3f})")
axes[0].scatter(fpr[np.argmin(np.abs(thresholds_roc - 0.45))],
                 tpr[np.argmin(np.abs(thresholds_roc - 0.45))],
                 color="black", zorder=5, s=100,
                 label="Threshold = 0.45")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
axes[0].set_title("ROC Curve: Random Forest @ 0.45 Threshold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Confusion Matrix for Random Forest at 0.45 Threshold
cm   = confusion_matrix(y_test, rand_for_pred_45)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Disease", "Disease"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix: Random Forest @ 0.45 Threshold")

# Performance Metrics for Random Forest at 0.45 Threshold
recall   = recall_score(y_test, rand_for_pred_45)
accuracy = accuracy_score(y_test, rand_for_pred_45)
f1       = f1_score(y_test, rand_for_pred_45)
precision = precision_score(y_test, rand_for_pred_45)

axes[1].set_xlabel(
    f"Recall: {recall:.3f}  |  Accuracy: {accuracy:.3f}  |  F1: {f1:.3f}  |  Precision: {precision:.3f}"
)

plt.suptitle("Random Forest: Optimal Threshold @ 0.45")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_random_forest_optimal_threshold.png",
             dpi=150, bbox_inches="tight")
plt.close()
print("Saving ROC Curve and Confusion Matrix plot as 08_random_forest_optimal_threshold.png..... Complete ✓")

# =============================================================================
# 6. SHAP Feature Importance
# =============================================================================

print("=" * 60)
print("6. SHAP Feature Importance")
print("=" * 60)

# --- [6.1] Compute SHAP Values ---
print("\n[6.1] Computing SHAP values for Random Forest...")

explainer   = shap.TreeExplainer(rand_for_model)
shap_values = explainer.shap_values(X_test_raw)

# Handle different shap_values structures
if isinstance(shap_values, list):
    shap_vals_disease = shap_values[1]       # old SHAP — list of arrays
elif shap_values.ndim == 3:
    shap_vals_disease = shap_values[:, :, 1] # new SHAP — 3D array, slice disease class
else:
    shap_vals_disease = shap_values          # already 2D

print(f"  shap_vals_disease shape : {shap_vals_disease.shape}")
print(f"  Test set shape          : {X_test_raw.shape}")
print(f"  Feature names length    : {len(feature_names)}")

# --- [6.2] SHAP Feature Importance Table ---
print("\n[6.2] SHAP Feature Importance Rankings:")
print("-" * 60)

mean_shap_vals = np.abs(shap_vals_disease).mean(axis=0)

# Debug
print(f"  mean_shap_vals shape : {mean_shap_vals.shape}")
print(f"  feature_names length : {len(feature_names)}")

shap_importance = pd.DataFrame({
    "Feature"     : feature_names,
    "Mean |SHAP|" : mean_shap_vals
})

shap_importance = shap_importance.sort_values("Mean |SHAP|",
                                               ascending=False)
shap_importance = shap_importance.reset_index(drop=True)
shap_importance.index += 1

print(shap_importance.to_string())

shap_importance.to_csv(f"{OUTPUT_DIR}/shap_importance.csv")
print(f"\n  Saved: shap_importance.csv")

# --- [6.3] SHAP Feature Importance Bar Plot ---
print("\n[6.3] Plotting SHAP Feature Importance...")

top_n     = 15
top_shap  = shap_importance.head(top_n)
top_feats = top_shap["Feature"].values
top_vals  = top_shap["Mean |SHAP|"].values

fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.barh(top_feats[::-1], top_vals[::-1],
                color="purple", edgecolor="black", alpha=0.8)

ax.set_xlabel("Mean |SHAP Value|")
ax.set_title(f"SHAP Feature Importance — Random Forest (Top {top_n})",
              fontweight="bold", fontsize=13)
ax.grid(axis="x", alpha=0.3)

for bar, val in zip(bars, top_vals[::-1]):
    ax.text(bar.get_width() + 0.001,
             bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_shap_importance.png",
             dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 09_shap_importance.png")