import os
from datetime import datetime, timedelta
import polars as pl

from fredapi import Fred

fred = Fred()

series = [
    "DFEDTARU", # Federal Funds Target Range - Upper Limit
    "DFEDTARL", # Federal Funds Target Range - Lower Limit
    "DTB3", # 3-Month Treasury Bill
    "DTB6", # 6-Month Treasury Bill
    "DTB1YR", # 1-Year Treasury Bill
    "DGS1", # 1-Year Treasury Constant Maturity
    "DGS2", # 2-Year Treasury Constant Maturity
    "DGS5", # 5-Year Treasury Constant Maturity
    "DGS10", # 10-Year Treasury Constant Maturity
    "DGS20", # 20-Year Treasury Constant Maturity
    "DGS30", # 30-Year Treasury Constant Maturity
    "DGS1MO", # 1-Month Treasury Constant Maturity
    "DGS3MO", # 3-Month Treasury Constant Maturity
    "DGS6MO", # 6-Month Treasury Constant Maturity
    "T5YIE", # 5-Year Breakeven Inflation Rate
    "T10YIE", # 10-Year Breakeven Inflation Rate
    "T20YIEM", # 20-Year Breakeven Inflation Rate
    "T30YIEM", # 30-Year Breakeven Inflation Rate
    "T5YIFR", # 5-Year, 5-Year Forward Inflation Rate
    "SP500", # S&P 500
    "NASDAQCOM", # NASDAQ Composite
]

data_dir = "data/fred"

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Define a start date (e.g., 1 year ago)
start_date = (datetime.now() - timedelta(days=1 * 365)).strftime("%Y-%m-%d")

# Function to fetch and save each series
def fetch_and_save_series(series_id):
    try:
        # Fetch the data from FRED with a start date
        data = fred.get_series(series_id, observation_start=start_date)
        
        # Convert the data to a Polars DataFrame
        df = pl.DataFrame({
            "date": data.index.to_list(),
            "value": data.values.tolist()
        })
        
        # Define the file path
        file_path = os.path.join(data_dir, f"{series_id}.parquet")
        
        # Check if the file already exists
        if os.path.exists(file_path):
            # Read the existing Parquet file
            existing_df = pl.read_parquet(file_path)
            
            # Concatenate the new data with the existing data
            combined_df = pl.concat([existing_df, df]).unique(subset=["date"])
            
            # Save the updated DataFrame to the Parquet file
            combined_df.write_parquet(file_path)
        else:
            # Save the DataFrame to a new Parquet file
            df.write_parquet(file_path)
        
        print(f"Processed {series_id}")
    except Exception as e:
        print(f"Error processing {series_id}: {e}")

def handle_fred(args):
    # Loop through each series and fetch/save the data
    for series_id in series:
        fetch_and_save_series(series_id)
        print(f"Processed {series_id}")

    print("All series have been processed and saved.")