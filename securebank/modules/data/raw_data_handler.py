import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class RawDataHandler:
    """
    Handles extraction, transformation, and description of the raw data for machine learning preprocessing.
    """

    def __init__(self, storage_path, save_path):
        """
        Initialize with paths to data storage and the save location.
        
        Parameters:
            storage_path (str or Path): The directory where raw data files are stored.
            save_path (str or Path): The directory to save cleaned data.
        """
        self.storage_path = Path(storage_path)
        self.save_path = Path(save_path)

    def extract(self, customer_information_filename, transaction_filename, fraud_information_filename):
        """
        Reads the raw data files and returns them as DataFrames.
        
        Parameters:
            customer_information_filename (str): The name of the .csv file.
            transaction_filename (str): The name of the .parquet file.
            fraud_information_filename (str): The name of the .json file.
        
        Returns:
            customer_information (pd.DataFrame): The .csv file data as a DataFrames.
            transaction_information (pd.DataFrame): The .parquet file data as a DataFrames.
            fraud_information (pd.DataFrame): The .json file data as a DataFrames.
        """
        customer_path = self.storage_path / customer_information_filename # Construct the full file paths.
        transaction_path = self.storage_path / transaction_filename
        fraud_path = self.storage_path / fraud_information_filename

        customer_information = pd.read_csv(customer_path) # Load the customer and transaction data.
        transaction_information = pd.read_parquet(transaction_path)

        with open(fraud_path, 'r') as f: # Load the fraud data from the .json file into a DataFrame.
            fraud_dict = json.load(f)

        fraud_information = pd.DataFrame({'is_fraud': fraud_dict})

        return customer_information, transaction_information, fraud_information

    def convert_dates(self, df):
        """
        Converts the 'trans_date_trans_time' column into separate time-based features.
        
        Parameters:
            df (pd.DataFrame): The new DataFrames with the added 'trans_date_trans_time' column.
        
        Returns:
            df (pd.DataFrame): The modified DataFrame with the new time-based columns.
        """
        if 'trans_date_trans_time' not in df.columns:
            raise ValueError("Expected column 'trans_date_trans_time' not found in DataFrame.")

        timestamps = pd.to_datetime(df['trans_date_trans_time']) # Convert the new column to datetime.

        df['day_of_week'] = timestamps.dt.day_name() # Extract various components from the timestamp.
        df['hour'] = timestamps.dt.hour
        df['minute'] = timestamps.dt.minute
        df['seconds'] = timestamps.dt.second
        df['day_date'] = timestamps.dt.day
        df['month_date'] = timestamps.dt.month_name()
        df['year_date'] = timestamps.dt.year

        df = df.drop(columns=['trans_date_trans_time']) # Drop the original timestamp column after decomposition is performed.

        return df

    def transform(self, customer_information, transaction_information, fraud_information):
        """
        Merges datasets, handles missing values, and removes any duplicates.
        
        Parameters:
            customer_information (pd.DataFrame): The .csv file data as a DataFrames.
            transaction_information (pd.DataFrame): The .parquet file data as a DataFrames.
            fraud_information (pd.DataFrame): The .json file data as a DataFrames.
        
        Returns:
            df (pd.DataFrame): The cleaned, merged DataFrame.
        """
        transaction_information = transaction_information.copy() # Copy the information o avoid modifying the original transaction data.
        transaction_information['is_fraud'] = fraud_information['is_fraud'] # Add fraud labels to the transaction data.

        df = pd.merge(transaction_information, customer_information, how='left', on='cc_num') # Merge the customer data on the credit card number.
        df = df.drop_duplicates() # Drop any duplicate entries.
        df = df.dropna() # Remove any records with missing values.

        return df

    def describe(self, raw_data):
        """
        Returns a summary dictionary of the data quality metrics.
        
        Parameters:
            raw_data (pd.DataFrame): The cleaned data.
        
        Returns:
            summary (dict): The summary metrics.
        """
        summary = {
            'number_of_records': len(raw_data), # The total number of rows.
            'number_of_columns': len(raw_data.columns), # The total number of features.
            'feature_names': list(raw_data.columns), # The column names.
            'number_missing_values': int(raw_data.isnull().sum().sum()), # The count of NaN's.
            'column_data_types': [str(dtype) for dtype in raw_data.dtypes] # The data types.
        }

        return summary