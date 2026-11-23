import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

def load_raw_data(customer_fp: str, transactions_fp: str, fraud_fp: str):
    """
    Load raw customer, transaction, and fraud label data from specified file paths.

    Parameters:
        customer_fp (str): File path to the customer CSV file.
        transactions_fp (str): File path to the transactions parquet file.
        fraud_fp (str): File path to the fraud JSON file.

    Returns:
        tuple:
            customers (pd.DataFrame): Customer information.
            transactions (pd.DataFrame): Transactions data.
            fraud_labels (pd.DataFrame): Fraud labels aligned to transactions index.
    """
    customers = pd.read_csv(customer_fp)
    transactions = pd.read_parquet(transactions_fp)

    fraud_labels = pd.read_json(fraud_fp, typ='series')
    fraud_labels.index = transactions.index # Align fraud labels index with transactions index and convert to DataFrame.
    fraud_labels = fraud_labels.rename('is_fraud').to_frame()

    return customers, transactions, fraud_labels

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth surface.

    Parameters:
        lat1, lon1, lat2, lon2 (float or array-like): Latitude and longitude of two points in degrees.

    Returns:
        float or np.ndarray: Distance in kilometers between the two points.
    """
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) # Convert decimal degrees to radians.
    
    dlon = (lon2 - lon1) # Compute differences.
    dlat = (lat2 - lat1) 
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2 # Haversine formula components.
    c = (2 * np.arcsin(np.sqrt(a)))
    r = 6371  # Radius of Earth in kilometers.

    return (c * r)

def compute_time_since_last_transaction(df: pd.DataFrame):
    """
    Compute the log-transformed time difference in hours since the last transaction per credit card number.

    Parameters:
        df (pd.DataFrame): Transactions DataFrame containing 'cc_num' and 'trans_date_trans_time' columns.

    Returns:
        pd.Series: Log(1 + time difference in hours) since the last transaction.
    """
    
    df_sorted = df.sort_values(['cc_num', 'trans_date_trans_time']) # Sort by credit card and transaction time to get chronological order.
    df_sorted['prev_trans_time'] = df_sorted.groupby('cc_num')['trans_date_trans_time'].shift(1) # Shift transaction times per card to get previous transaction time.
    
    time_diff = (df_sorted['trans_date_trans_time'] - df_sorted['prev_trans_time']).dt.total_seconds() / 3600 # Calculate time difference in hours.
    time_diff = time_diff.fillna(0) # Fill NaN (for first transaction) with zero.
    
    return np.log1p(time_diff) # Return log(1 + time_diff) to reduce skew.

def compute_merchant_category_fraud_rate(df: pd.DataFrame, alpha=1, beta=2):
    """
    Compute smoothed fraud rate per merchant category using a Beta prior smoothing.

    Parameters:
        df (pd.DataFrame): Transactions DataFrame containing 'category' and 'is_fraud' columns.
        alpha (int, optional): Smoothing parameter numerator. Default is 1.
        beta (int, optional): Smoothing parameter denominator offset. Default is 2.

    Returns:
        pd.Series: Fraud rate indexed by merchant category.
    """
    
    fraud_counts = df.groupby('category')['is_fraud'].sum() # Count fraud occurrences and total transactions per category.
    total_counts = df.groupby('category')['is_fraud'].count()
    
    fraud_rate = (fraud_counts + alpha) / (total_counts + alpha + beta) # Compute smoothed fraud rate to avoid zero rates.

    return fraud_rate

def engineer_features(customers: pd.DataFrame, transactions: pd.DataFrame, fraud_labels: pd.DataFrame):
    """
    Engineer features for fraud detection by merging customer, transaction, and fraud data.

    Parameters:
        customers (pd.DataFrame): Customer information DataFrame.
        transactions (pd.DataFrame): Transactions DataFrame.
        fraud_labels (pd.DataFrame): Fraud labels DataFrame with 'is_fraud' column.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame ready for modeling.
    """
    df = transactions.copy() # Work on a copy to avoid mutating original transactions.
    df["is_fraud"] = fraud_labels["is_fraud"] # Add fraud label column.
    
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"]) # Convert transaction time to datetime and extract temporal features.
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    
    print("Computing time since last transaction feature...")
    df['time_since_last_trans'] = compute_time_since_last_transaction(df) # Compute time since last transaction feature.

    print("Computing merchant category fraud rate feature...")
    fraud_rate_map = compute_merchant_category_fraud_rate(df) # Compute merchant category fraud rate and map it.

    df['merchant_cat_fraud_rate'] = df['category'].map(fraud_rate_map)
    df['merchant_cat_fraud_rate'] = df['merchant_cat_fraud_rate'].fillna(df['is_fraud'].mean()) # Fill missing fraud rates with overall fraud mean.

    df = df.merge(customers, on="cc_num", how="left") # Merge customer info into transaction DataFrame.
    
    df["dob"] = pd.to_datetime(df["dob"], errors='coerce') # Calculate age from date of birth and transaction date.
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    ) # Calculate distance between customer and merchant locations.
    
    df = pd.get_dummies(df, columns=["category"], prefix="cat") # One-hot encode merchant categories.
    
    drop_cols = [
        "dob", "unix_time", "first", "last", "street",
        "city", "state", "zip", "lat", "long", "merch_lat", "merch_long", "merchant"
    ] # Drop columns not needed for modeling.
    
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    df = df.rename(columns={"trans_date_trans_time": "trans_date"}) # Rename transaction time column for clarity.

    print("Columns in engineered DataFrame:", df.columns.tolist()) # Print final columns for debugging.

    return df

def save_dataset(df: pd.DataFrame, output_fp: str):
    """
    Save DataFrame to CSV at the specified file path.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_fp (str): Output file path for CSV.
    """
    os.makedirs(os.path.dirname(output_fp), exist_ok=True) # Create directory if it doesn't exist.
    
    df.to_csv(output_fp, index=False) # Save to CSV without row indices.
    print(f"Dataset saved to {output_fp}")

def build_dataset():
    """
    Build the complete dataset by loading raw data, engineering features, performing a temporal split,
    and saving full, train, and test datasets to disk.

    Returns:
        str: File path of the full dataset CSV.
    """
    print("Loading raw data...")

    customer_fp = "securebank/data_sources/customer_release.csv"
    transactions_fp = "securebank/data_sources/transactions_release.parquet"
    fraud_fp = "securebank/data_sources/fraud_release.json"

    customers, transactions, fraud_labels = load_raw_data(
        customer_fp, transactions_fp, fraud_fp
    ) # Load raw input data.

    print("Engineering features...")
    df = engineer_features(customers, transactions, fraud_labels)

    print("Performing temporal customer split...")
    df["trans_date"] = pd.to_datetime(df["trans_date"]) # Convert to datetime for splitting.

    first_seen = df.groupby("cc_num")["trans_date"].min().rename("first_seen") # Compute first transaction date per customer.
    df = df.merge(first_seen, on="cc_num", how="left")

    df["cutoff"] = df["first_seen"] + timedelta(days=180) # Define cutoff date as 180 days after first transaction per customer.

    train_df = df[df["trans_date"] <= df["cutoff"]].drop(columns=["first_seen", "cutoff"]) # Split into train and test sets based on cutoff date.
    test_df = df[df["trans_date"] > df["cutoff"]].drop(columns=["first_seen", "cutoff"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Prepare output directory and file paths with timestamp.
    base_dir = "securebank/storage/datasets"
    os.makedirs(base_dir, exist_ok=True)

    full_fp = f"{base_dir}/dataset_full_{timestamp}.csv"
    train_fp = f"{base_dir}/dataset_train_{timestamp}.csv"
    test_fp = f"{base_dir}/dataset_test_{timestamp}.csv"

    save_dataset(df, full_fp) # Save datasets to CSV.
    save_dataset(train_df, train_fp)
    save_dataset(test_df, test_fp)

    print(f"Train set size: {train_df.shape}") # Print dataset sizes for confirmation.
    print(f"Test set size:  {test_df.shape}")
    print("All datasets saved.")

    return full_fp

def process_input_for_prediction(input_data: dict):
    """
    Process a single input dictionary of transaction data into a feature DataFrame
    ready for prediction by the fraud model.

    Parameters:
        input_data (dict): Raw input transaction data with expected keys.

    Returns:
        pd.DataFrame: Processed single-row DataFrame with engineered features.
    """
    df = pd.DataFrame([input_data])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    
    df["time_since_last_trans"] = 0 # Assign default values for features that require historical data or statistics.
    df["merchant_cat_fraud_rate"] = 0.01
    df["age"] = 40
    df["distance_km"] = 0
    
    categories = [
        "cat_FOOD_DINING", "cat_grocery_pos", "cat_personal_care",
        "cat_shopping_net", "cat_entertainment", "cat_gas_transport"
    ] # Initialize known category columns to zero.

    for cat_col in categories:
        df[cat_col] = 0
    
    cat_col_name = f"cat_{input_data['category']}" # Set the column corresponding to the input transaction category to 1.

    if cat_col_name in df.columns:
        df[cat_col_name] = 1

    drop_cols = ["trans_date_trans_time", "merchant", "category", "cc_num", "unix_time", "merch_lat", "merch_long"] # Drop columns irrelevant for prediction model input.
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def main():
    """
    Main test function to build the dataset and display summary information.
    """
    print("Starting dataset_builder.py test run...")

    try:
        output_fp = build_dataset()
        print(f"Dataset built successfully and saved at: {output_fp}")
        df = pd.read_csv(output_fp)
        print(f"Loaded dataset shape: {df.shape}")
        print(f"Sample columns: {list(df.columns)[:10]}")
        print(f"Sample data preview:\n{df.head(3)}")

    except Exception as e:
        print(f"Error during dataset build: {e}")

if __name__ == "__main__":
    main()