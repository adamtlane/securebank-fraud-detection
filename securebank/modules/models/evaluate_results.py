import os
import sys
import glob
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Define base directories for consistent file path resolution across the project.
MODELS_DIR = os.path.join(BASE_DIR, "storage", "models")
DATASETS_DIR = os.path.join(BASE_DIR, "storage", "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "storage", "results")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib") # Preprocessor file path assumed to be constant.

def get_latest_file(directory: str, pattern: str):
    """
    Find the latest file matching the given glob pattern in a directory.

    Parameters:
        directory (str): Directory path to search within.
        pattern (str): Glob pattern to match files.

    Returns:
        str or None: Full path to the latest matching file by modification time,
                     or None if no files matched.
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:

        return None
    
    return max(files, key=os.path.getmtime)

def load_latest_model():
    """
    Load the most recent model file (.joblib) in the models directory,
    excluding any preprocessor files.

    Returns:
        str or None: Path to the latest model file or None if none found.
    """
    model_files = [
        f for f in glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
        if not os.path.basename(f).startswith("preprocessor")
    ] # List all .joblib files excluding those starting with 'preprocessor'.

    if not model_files:
        return None

    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Using latest model: {os.path.basename(latest_model)}")

    return latest_model

def load_latest_dataset():
    """
    Load the latest full dataset CSV from the datasets directory.

    Returns:
        str or None: Path to the latest dataset file or None if none found.
    """
    latest_dataset = get_latest_file(DATASETS_DIR, "dataset_full_*.csv")

    if latest_dataset:
        print(f"Using latest full dataset: {os.path.basename(latest_dataset)}")

    return latest_dataset

def save_confusion_matrix(y_true, y_pred, path):
    """
    Generate and save a confusion matrix heatmap plot.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        path (str): File path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_precision_recall_curve(y_true, y_scores, path):
    """
    Generate and save a precision-recall curve plot.

    Parameters:
        y_true (array-like): True class labels.
        y_scores (array-like): Predicted probabilities or scores.
        path (str): File path to save the precision-recall curve plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(16, 9))
    plt.plot(recall, precision, marker=".")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_roc_curve(y_true, y_scores, path):
    """
    Generate and save a ROC curve plot.

    Parameters:
        y_true (array-like): True class labels.
        y_scores (array-like): Predicted probabilities or scores.
        path (str): File path to save the ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(16, 9))
    plt.plot(fpr, tpr, marker=".", label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for random classifier.
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    """
    Main routine to evaluate the fraud detection model:
    - Loads latest or specified model and dataset.
    - Loads the preprocessor to transform input features.
    - Cleans and prepares the dataset consistent with training preprocessing.
    - Runs model predictions.
    - Generates classification report and confusion matrix.
    - Saves numerical results and visualizations to results directory.
    """
    print(f"BASE_DIR: {BASE_DIR}") # Print resolved directories for debugging.
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"DATASETS_DIR: {DATASETS_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists.

    if len(sys.argv) == 3: # Determine model and dataset paths from CLI args or by auto-discovery.
        model_path = sys.argv[1]
        data_path = sys.argv[2]
    else:
        model_path = load_latest_model()
        data_path = load_latest_dataset()

        if not model_path or not data_path:
            print("No model or dataset found. Provide paths manually:\n"
                  "Usage: python evaluate_results.py <model_path> <dataset_path>")
            sys.exit(1)
    
    print("Loading model...") # Load the model from file.
    model = joblib.load(model_path)
    
    print("Loading preprocessor...") # Load the preprocessor from file.

    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        sys.exit(1)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Loading dataset...") # Load dataset as DataFrame.
    df = pd.read_csv(data_path)

    if 'is_fraud' not in df.columns: # Check that target column exists.
        print("Dataset missing 'is_fraud' column.")
        sys.exit(1)

    print("Dropping rows with missing 'is_fraud' values...") # Drop rows missing target values to maintain consistency with training data.
    before_rows = len(df)
    df = df.dropna(subset=["is_fraud"])
    dropped = before_rows - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} rows with missing 'is_fraud' values.")

    if 'trans_date' in df.columns: # Feature engineering for datetime fields if present.
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        df['trans_month'] = df['trans_date'].dt.month
        df['trans_day'] = df['trans_date'].dt.day
        df['trans_dayofweek'] = df['trans_date'].dt.dayofweek
        df = df.drop(columns=['trans_date'])  # Drop original column to avoid duplication
    
    drop_cols = [
        "trans_num", "cc_num", "first", "last", "street", "city", "state", "zip",
        "job", "dob", "merchant", "trans_date_trans_time", "unix_time",
        "fraud_id", "index_x", "index_y"
    ] # Drop columns that were removed during training preprocessing.

    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    X = df.drop(columns=['is_fraud']) # Separate features and target variable.
    y = df['is_fraud']

    # Evaluate on the same held out 30% test split that train_model.py uses, so the
    # transactions the model was trained on are not counted in the results.
    _, X, _, y = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Evaluating on the held out test split: {len(y)} transactions")
    
    print("Transforming features with preprocessor...") # Transform raw features using the loaded preprocessor.
    X_processed = preprocessor.transform(X)
    
    print("Running predictions...") # Run prediction using the loaded model.
    y_pred = model.predict(X_processed)
    
    if hasattr(model, "predict_proba"): # Attempt to get probability scores if model supports it, else fallback.
        y_scores = model.predict_proba(X_processed)[:, 1]
    else:
        y_scores = y_pred

    print("Classification Report:") # Generate and print classification report (precision, recall, f1-score).
    report_dict = classification_report(y, y_pred, output_dict=True)
    print(classification_report(y, y_pred))

    cm_array = confusion_matrix(y, y_pred) # Generate and print confusion matrix.
    print("Confusion Matrix:")
    print(cm_array)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Save metrics and plots with timestamp for traceability.
    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    results_data = {
        "classification_report": report_dict,
        "confusion_matrix": cm_array.tolist(),
        "roc_auc": roc_auc_score(y, y_scores) if len(set(y)) > 1 else None
    }

    with open(json_path, "w") as f: # Save numerical results to JSON file.
        json.dump(results_data, f, indent=4)

    save_confusion_matrix(y, y_pred, os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png")) # Save visual plots of evaluation metrics.
    save_precision_recall_curve(y, y_scores, os.path.join(RESULTS_DIR, f"precision_recall_{timestamp}.png"))
    save_roc_curve(y, y_scores, os.path.join(RESULTS_DIR, f"roc_curve_{timestamp}.png"))

    print(f"Numerical results saved to {json_path}") # Print paths where results are saved.
    print(f"Plots saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()