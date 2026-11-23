import pandas as pd
import joblib
import os
import sys
import glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Dynamically resolve paths relative to this script's location.
DATASETS_DIR = os.path.join(BASE_DIR, "storage", "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "storage", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

def get_latest_full_dataset():
    """
    Find the most recent full dataset CSV file in the datasets directory.

    Returns:
        str: File path of the latest full dataset CSV.

    Raises:
        FileNotFoundError: If no full dataset CSV files are found.
    """
    pattern = os.path.join(DATASETS_DIR, "*full*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No full dataset CSV found in {DATASETS_DIR}")
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"Auto-detected latest full dataset: {latest_file}")

    return latest_file

def load_data(data_path: str):
    """
    Load dataset from CSV or parquet file and ensure 'is_fraud' target column exists.

    Parameters:
        data_path (str): File path to dataset (CSV or parquet).

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        ValueError: If target column 'is_fraud' is not found in dataset.
    """
    print(f"Loading data from {data_path}...")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")

    if "label" in df.columns and "is_fraud" not in df.columns: # Rename 'label' column to 'is_fraud' if necessary.
        print("Renaming 'label' column to 'is_fraud'...")
        df.rename(columns={"label": "is_fraud"}, inplace=True)

    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' not found in dataset.")

    return df

def preprocess_data(df: pd.DataFrame):
    """
    Clean dataframe by dropping rows with missing target, removing unnecessary columns,
    extracting date features, and separating features and target.

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        tuple: (X, y)
            X (pd.DataFrame): Features dataframe.
            y (pd.Series): Target labels as integers.
    """
    print("Preprocessing data...")

    before_rows = len(df)
    df = df.dropna(subset=["is_fraud"])
    dropped = before_rows - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} rows with missing 'is_fraud' values.")

    drop_cols = [
        "trans_num", "cc_num", "first", "last", "street", "city", "state", "zip",
        "job", "dob", "merchant", "trans_date_trans_time", "unix_time",
        "fraud_id", "index_x", "index_y"
    ] # Columns to drop as irrelevant or redundant.

    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    if "ccity_pop" in df.columns and "city_pop" not in df.columns: # Fix known typo in column name if present.
        df.rename(columns={"ccity_pop": "city_pop"}, inplace=True)

    if "trans_date" in df.columns: # Extract date/time features from 'trans_date' column.
        try:
            df["trans_date"] = pd.to_datetime(df["trans_date"], errors='coerce')
            df["trans_month"] = df["trans_date"].dt.month.astype("Int64")
            df["trans_day"] = df["trans_date"].dt.day.astype("Int64")
            df["trans_dayofweek"] = df["trans_date"].dt.dayofweek.astype("Int64")
            
            df.drop(columns=["trans_date"], inplace=True) # Drop original date column to avoid large one-hot expansions.
            print("Extracted 'trans_month', 'trans_day', 'trans_dayofweek' from 'trans_date'.")

        except Exception as e:
            print(f"Failed to parse 'trans_date': {e}")
            df.drop(columns=["trans_date"], inplace=True)

    X = df.drop(columns=["is_fraud"], errors="ignore") # Separate features and target variable.
    y = df["is_fraud"].astype(int)

    print(f"Features shape before processing: {X.shape}")
    print("Target distribution:")
    print(y.value_counts(dropna=False))

    return X, y

def build_preprocessor(X: pd.DataFrame, max_categories: int = 20):
    """
    Build a ColumnTransformer pipeline to preprocess numeric and categorical features,
    using cardinality-based encoding for categorical variables.

    Parameters:
        X (pd.DataFrame): Features dataframe.
        max_categories (int): Threshold to split categorical columns by cardinality.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist() # Identify categorical and numeric columns.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    low_card_cols = [col for col in categorical_cols if X[col].nunique() <= max_categories] # Split categorical columns by cardinality threshold.
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > max_categories]

    print(f"Low cardinality categorical columns (<= {max_categories} categories): {low_card_cols}")
    print(f"High cardinality categorical columns (> {max_categories} categories): {high_card_cols}")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]) # Numeric pipeline: median imputation + scaling.
    
    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]) # Low-cardinality categorical pipeline: constant imputation + one-hot encoding.

    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]) # High-cardinality categorical pipeline: constant imputation + ordinal encoding.

    transformers = [
        ('num', numeric_transformer, numeric_cols),
    ]

    if low_card_cols:
        transformers.append(('low_card_cat', low_card_transformer, low_card_cols))
    if high_card_cols:
        transformers.append(('high_card_cat', high_card_transformer, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3) # Build ColumnTransformer with specified pipelines.

    return preprocessor

def train_and_save_model(data_path: str):
    """
    Complete training pipeline:
    - Load and preprocess data
    - Split train/test
    - Build and fit preprocessor
    - Train RandomForest model
    - Evaluate and print metrics
    - Save model and preprocessor with timestamp and latest pointers

    Parameters:
        data_path (str): Path to dataset file.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score for fraud class.
    """
    df = load_data(data_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    ) # Split dataset into training and test sets with stratification on target.

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    preprocessor = build_preprocessor(X_train) # Build and fit preprocessor on training data only.

    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed feature shape (train): {X_train_processed.shape}")
    print(f"Processed feature shape (test): {X_test_processed.shape}")

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Initialize and train Random Forest classifier.
    model.fit(X_train_processed, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_processed) # Predict and evaluate on test set.
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fraud class precision: {report['1']['precision']:.4f}")
    print(f"Fraud class recall:    {report['1']['recall']:.4f}")
    print(f"Fraud class F1-score:  {report['1']['f1-score']:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Ensure models directory exists and save model & preprocessor with timestamp.

    model_path_timestamped = os.path.join(MODELS_DIR, f"rf_model_{timestamp}.joblib")
    preprocessor_path_timestamped = os.path.join(MODELS_DIR, f"preprocessor_{timestamp}.joblib")

    print(f"Saving model to {model_path_timestamped}...")
    joblib.dump(model, model_path_timestamped)

    print(f"Saving preprocessor to {preprocessor_path_timestamped}...")
    joblib.dump(preprocessor, preprocessor_path_timestamped)

    joblib.dump(model, MODEL_PATH) # Save/update latest model and preprocessor pointers.
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Model and preprocessor saved with latest pointers.")

    return {
        "accuracy": accuracy,
        "precision_fraud": report['1']['precision'],
        "recall_fraud": report['1']['recall'],
        "f1_fraud": report['1']['f1-score']
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No dataset path provided, auto-detecting latest full dataset...")
        data_path = get_latest_full_dataset()
    else:
        data_path = sys.argv[1]

    try:
        train_and_save_model(data_path)
    except Exception as e:
        print(f"Error: {e}")