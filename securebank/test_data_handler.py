from modules.data.raw_data_handler import RawDataHandler

handler = RawDataHandler(storage_path="data_sources", save_path="cleaned") # Initialize the RawDataHandler file.

customer_info, transaction_info, fraud_info = handler.extract(
    "customer_release.csv", "transactions_release.parquet", "fraud_release.json"
) # Extract the raw data files.

cleaned_data = handler.transform(customer_info, transaction_info, fraud_info) # Transform the data.
cleaned_data = handler.convert_dates(cleaned_data)

description = handler.describe(cleaned_data) # Describe and print the cleaned data.
print(description)