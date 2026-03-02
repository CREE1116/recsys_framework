import pandas as pd
import numpy as np
import os
import json

def verify_dataset(file_path):
    print(f"--- Verifying {os.path.basename(file_path)} ---")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 1. Load data
    df = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'timestamp', 'lat', 'lon', 'item_id'])
    
    # 2. Check for exact duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate interactions: {duplicates}")
    
    # 3. Check for User-Item duplicates (ignoring timestamp/lat/lon)
    ui_duplicates = df.duplicated(subset=['user_id', 'item_id']).sum()
    print(f"User-Item duplicates (multiple interactions): {ui_duplicates}")
    
    # 4. Basic stats
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_inter = len(df)
    density = n_inter / (n_users * n_items)
    
    print(f"Interactions: {n_inter}")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Density: {density:.6f}")
    
    # 5. Check temporal ordering (if possible)
    if 'timestamp' in df.columns:
        # Check if users have multiple interactions with same item at different times
        ui_time_counts = df.groupby(['user_id', 'item_id']).size()
        repeats = (ui_time_counts > 1).sum()
        print(f"Users interacting with same item multiple times: {repeats}")

def check_data_loader_split():
    print("\n--- Checking Data Loader Split (Potential Leakage) ---")
    # This is a conceptual check. We can't easily run the full data loader here without setup,
    # but we can check if the subset was created correctly.
    subset_path = "/Users/leejongmin/code/recsys_framework/data/gowalla/gowalla_subset.txt"
    if os.path.exists(subset_path):
        df = pd.read_csv(subset_path, sep='\t', header=None, names=['user_id', 'timestamp', 'lat', 'lon', 'item_id'])
        
        # Check if timestamps are all identical or suspicious
        unique_timestamps = df['timestamp'].nunique()
        print(f"Unique timestamps: {unique_timestamps}")
        if unique_timestamps == 1:
            print("WARNING: All timestamps are identical! Temporal split will be random/unstable.")
        
        # Check timestamp range
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    verify_dataset("/Users/leejongmin/code/recsys_framework/data/gowalla/gowalla_subset.txt")
    check_data_loader_split()
