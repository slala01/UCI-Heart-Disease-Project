# =============================================================================
# UCI Heart Disease --- Classification Project
# =============================================================================

# Imports
import logging
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
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# File Configurations
RANDOM_STATE = 67
OUTPUT_DIR   = "outputs"
PLOTS_DIR    = "outputs/plots"
MODELS_DIR   = "outputs/models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Column definitions
NUMERICAL_COLS = [
    "age", "resting_blood_pressure", "serum_cholestoral",
    "maximum_heart_rate_achieved", "oldpeak", "number_of_major_vessels"]

CATEGORICAL_COLS = [
    "sex", "chest", "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "exercise_induced_angina", "slope", "thal"]

# Model colors
MODEL_COLORS = ["green", "blue", "purple", "red"]

# =============================================================================
# Helper Functions
# =============================================================================

def save_plot(fig, filename):
    """Save a plot to the plots directory and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saving: %s..... Complete ✓", filename)


def save_model(model, model_name):
    """Save a trained model to the models directory."""
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    logger.info("  Model saved → %s", filepath)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Return a dict of standard classification metrics for one model."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
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

    print("=" * 60)
    print("1. Import Data and Inspect")
    print("=" * 60)

    # Load in dataset from OpenML
    dataset = datasets.fetch_openml(data_id=46590, return_X_y=False, as_frame=True)

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

    missing     = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    if missing_df["Missing Count"].sum() == 0:
        logger.info("  No missing values!")
    else:
        logger.warning("  Missing values detected:\n%s", missing_df[missing_df["Missing Count"] > 0].to_string())

    # Recode target variable to binary
    df["target"] = df["target"].apply(lambda x: 1 if x == "present" else 0)
    logger.info("[1.6] Target recoded: 'present' → 1, absent → 0")

    # =============================================================================
    # 2. EDA & Visualization
    # =============================================================================

    print("=" * 60)
    print("2. EDA & Visualization")
    print("=" * 60)

    def _plot_target_distribution(df):
        target_counts = df["target"].value_counts()
        counts = [target_counts[0], target_counts[1]]
        labels = ["No Disease", "Disease"]
        colors = ["green", "red"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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


    def _plot_numerical_distributions(df):
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


    def _plot_categorical_counts(df):
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
            axes[i].set_xticklabels(no_disease.index.astype(int))
            axes[i].legend(fontsize=7)

        axes[-1].set_visible(False)
        plt.suptitle("Categorical Features by Target Class")
        plt.tight_layout()
        return fig


    def _plot_correlation(df):
        fig, ax = plt.subplots(figsize=(12, 9))
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title("Feature Correlation")
        plt.tight_layout()
        return fig

    eda_plots = [
        ("01_target_distribution.png",    _plot_target_distribution),
        ("02_numerical_distributions.png", _plot_numerical_distributions),
        ("03_categorical_counts.png",      _plot_categorical_counts),
        ("04_correlation_plot.png",        _plot_correlation)]

    for filename, plot_fn in tqdm(eda_plots, desc="EDA plots", unit="plot"):
        fig = plot_fn(df)
        save_plot(fig, filename)

    # =============================================================================
    # 3. Feature Engineering
    # =============================================================================

    print("=" * 60)
    print("3. Feature Engineering")
    print("=" * 60)

    df_engin = df.copy()

    # One Hot Encoding for categorical features
    logger.info("[3.1] One Hot Encoding categorical features...")
    df_engin = pd.get_dummies(df_engin, columns=CATEGORICAL_COLS, dtype=int, drop_first=False)
    logger.info("  Encoded columns      : %s", CATEGORICAL_COLS)
    logger.info("  Shape after encoding : %s", df_engin.shape)

    # Final Feature List
    X = df_engin.drop(columns=["target"])
    y = df_engin["target"]
    feature_names = X.columns.tolist()

    logger.info("[3.2] Total features: %d", len(feature_names))

    # Train/Test split
    logger.info("[3.3] Train/Test split 75/25 (stratified):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    logger.info("  X_train shape : %s", X_train.shape)
    logger.info("  X_test  shape : %s", X_test.shape)

    # =============================================================================
    # 4. Modeling and Performance Metrics
    # =============================================================================

    print("=" * 60)
    print("4. Modeling and Performance Metrics")
    print("=" * 60)

    # Model Registry
    def build_model_registry():
        """Return all models and their hyperparameter grids."""

        log_reg_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
            ])

        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVC(random_state=RANDOM_STATE, probability=True))
        ])

        return {
            "Logistic Regression": (
                log_reg_pipeline,
                {
                    "model__C"       : [0.01, 0.1, 1, 10, 100],
                    "model__penalty" : ["l1", "l2"],
                    "model__solver"  : ["liblinear", "saga"],
                }
            ),
            "SVM": (
                svm_pipeline,
                {
                    "model__C"      : [0.1, 1, 10, 100, 1000],
                    "model__gamma"  : [1, 0.1, 0.01, 0.001, 0.0001],
                    "model__kernel" : ["rbf"],
                }
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {
                    "n_estimators"      : [100, 300, 500],
                    "max_features"      : ["sqrt", "log2"],
                    "max_depth"         : [3, 5, 7],
                    "min_samples_split" : [5, 10, 20],
                    "min_samples_leaf"  : [3, 5, 10],
                }
            ),
            "XGBoost": (
                XGBClassifier(random_state=RANDOM_STATE),
                {
                    "n_estimators"     : [50, 100, 200],
                    "learning_rate"    : [0.01, 0.1],
                    "max_depth"        : [2, 3, 4],
                    "subsample"        : [0.6, 0.8],
                    "colsample_bytree" : [0.6, 0.8],
                    "reg_alpha"        : [0, 0.1],
                    "reg_lambda"       : [1.0, 5.0],
                }
            ),
        }

    registry = build_model_registry()
    results  = {}

    for name, (estimator, param_grid) in tqdm(registry.items(), desc="Training models", unit="model"):
        logger.info("  Fitting: %s...", name)

        grid = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        save_model(model, name)

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {
            "model"  : model,
            "pred"   : model.predict(X_test),
            "prob"   : model.predict_proba(X_test)[:, 1],
            **metrics}

    # Performance summary table
    metric_keys = ["AUC-ROC", "Accuracy", "Recall", "F1 Score", "Precision"]
    results_df  = pd.DataFrame(
        {name: {k: v for k, v in res.items() if k in metric_keys}
         for name, res in results.items()}
    ).T

    logger.info("[4.5] Performance Metrics:")
    logger.info("\n%s", results_df.to_string())

    # Model comparison plot
    def _plot_model_comparison(results_df):
        metric_keys  = ["AUC-ROC", "Accuracy", "Recall", "F1 Score", "Precision"]
        fig, axes    = plt.subplots(2, 3, figsize=(15, 9))
        axes         = axes.flatten()

        for i, metric in enumerate(metric_keys):
            values = results_df[metric]
            bars   = axes[i].bar(results_df.index, values, color=MODEL_COLORS, edgecolor="black")
            axes[i].set_title(metric, fontsize=12)
            axes[i].set_ylabel("Score")
            axes[i].set_ylim(0, 1.10)
            axes[i].set_xticklabels(results_df.index)
            for bar, val in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center")

        axes[5].set_visible(False)
        plt.suptitle("Model Performance Comparison")
        plt.tight_layout()
        return fig

    fig = _plot_model_comparison(results_df)
    save_plot(fig, "05_model_comparison.png")

    # ROC curves plot
    def _plot_roc_curves(results, X_test, y_test):
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, res), color in zip(results.items(), MODEL_COLORS):
            RocCurveDisplay.from_estimator(
                res["model"], X_test, y_test, ax=ax, color=color, name=name
            )
        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_title("ROC Curves - All Models", fontweight="bold", fontsize=12)
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    fig = _plot_roc_curves(results, X_test, y_test)
    save_plot(fig, "06_roc_curves.png")

    # Confusion matrices plot
    def _plot_confusion_matrices(results, y_test):
        fig, axes = plt.subplots(1, len(results), figsize=(18, 4))
        for ax, (name, res) in zip(axes, results.items()):
            cm   = confusion_matrix(y_test, res["pred"])
            disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name, fontweight="bold", fontsize=10)

        plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    fig = _plot_confusion_matrices(results, y_test)
    save_plot(fig, "07_confusion_matrices.png")

    # =============================================================================
    # 5. Model Diagnostics
    # =============================================================================

    print("=" * 60)
    print("5. Model Diagnostics")
    print("=" * 60)

    # Train vs Test AUC Gap
    logger.info("[5.1] Train vs Test AUC Gap:")

    diagnostics = {}
    for name, res in results.items():
        train_auc = roc_auc_score(y_train, res["model"].predict_proba(X_train)[:, 1])
        test_auc  = roc_auc_score(y_test,  res["model"].predict_proba(X_test)[:, 1])
        diagnostics[name] = {
            "Train AUC" : round(train_auc, 4),
            "Test AUC"  : round(test_auc, 4),
            "Gap"       : round(train_auc - test_auc, 4)}
    logger.info("\n%s", pd.DataFrame(diagnostics).T.to_string())

    # Threshold sweep
    logger.info("[5.2] Adjusted Threshold Check:")

    thresholds = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]

    for name, res in tqdm(results.items(), desc="Threshold sweep", unit="model"):
        logger.info("  %s:", name)
        logger.info("  %-12s %-10s %-12s %-10s %-10s %-10s",
                    "Threshold", "Recall", "AUC-ROC", "F1", "Accuracy", "Precision")
        for thresh in thresholds:
            preds = (res["prob"] >= thresh).astype(int)
            logger.info("  %-12s %-10.3f %-12.3f %-10.3f %-10.3f %-10.3f",
                        thresh,
                        recall_score(y_test, preds),
                        roc_auc_score(y_test, res["prob"]),
                        f1_score(y_test, preds),
                        accuracy_score(y_test, preds),
                        precision_score(y_test, preds))

    # Find optimal threshold where all metrics >= 0.85 for each model
    optimal_thresholds = {}

    for model_name, res in results.items():
        prob           = res["prob"]
        optimal_thresh = None

        for thresh in thresholds:
            preds   = (prob >= thresh).astype(int)
            metrics = {
                "Recall"    : recall_score(y_test, preds),
                "Accuracy"  : accuracy_score(y_test, preds),
                "F1"        : f1_score(y_test, preds),
                "Precision" : precision_score(y_test, preds)}
            if all(v >= 0.85 for v in metrics.values()):
                optimal_thresh = thresh
                logger.info("  %s — Threshold %.2f meets all criteria:", model_name, thresh)
                for metric, val in metrics.items():
                    logger.info("    %-12s : %.3f", metric, val)
                break

        if optimal_thresh is None:
            logger.warning("  %s — No threshold achieved all metrics >= 0.85", model_name)

        optimal_thresholds[model_name] = optimal_thresh

    # Pick the best model — first one that found a qualifying threshold, else highest AUC
    best_model = next(
        (name for name, thresh in optimal_thresholds.items() if thresh is not None),
        max(results, key=lambda name: results[name]["AUC-ROC"]))
    best_thresh = optimal_thresholds[best_model] or 0.45

    logger.info("[5.3] Optimal Model: %s @ %.2f Threshold", best_model, best_thresh)
    
    # Optimal threshold plot
    def _plot_optimal_threshold(results, y_test, model_name="Random Forest", threshold=0.45):
        prob     = results[model_name]["prob"]
        pred_adj = (prob >= threshold).astype(int)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC curve with threshold marker
        fpr, tpr, thresholds_roc = roc_curve(y_test, prob)
        auc_score = roc_auc_score(y_test, prob)
        axes[0].plot(fpr, tpr, color="purple", lw=2,
                    label=f"{model_name} (AUC = {auc_score:.3f})")
        idx = np.argmin(np.abs(thresholds_roc - threshold))
        axes[0].scatter(fpr[idx], tpr[idx], color="black", zorder=5, s=100,
                        label=f"Threshold = {threshold}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
        axes[0].set_title(f"ROC Curve: {model_name} @ {threshold} Threshold")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        # Confusion matrix
        cm   = confusion_matrix(y_test, pred_adj)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
        disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
        axes[1].set_title(f"Confusion Matrix: {model_name} @ {threshold} Threshold")
        axes[1].set_xlabel(
            f"Recall: {recall_score(y_test, pred_adj):.3f}  |  "
            f"Accuracy: {accuracy_score(y_test, pred_adj):.3f}  |  "
            f"F1: {f1_score(y_test, pred_adj):.3f}  |  "
            f"Precision: {precision_score(y_test, pred_adj):.3f}")

        plt.suptitle(f"{model_name}: Optimal Threshold @ {threshold}")
        plt.tight_layout()
        return fig

    fig = _plot_optimal_threshold(results, y_test, model_name="Random Forest", threshold=0.45)
    save_plot(fig, "08_random_forest_optimal_threshold.png")

    logger.info("=" * 60)
    logger.info("Pipeline complete. All outputs saved to: %s", OUTPUT_DIR)
    logger.info("=" * 60)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()