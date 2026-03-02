import pandas as pd
import sys

input_file = "/Users/leejongmin/code/recsys_framework/data/gowalla/loc-gowalla_totalCheckins.txt"
output_file = "/Users/leejongmin/code/recsys_framework/data/gowalla/gowalla.csv"

def convert():
    print(f"Reading {input_file}...")
    # The format is user, time, lat, lng, loc_id
    # Likely tab separated or space separated with tabs in between
    df = pd.read_csv(input_file, sep='\t', header=None, names=['user', 'time', 'latitude', 'longitude', 'location_id'])
    
    print("Selecting relevant columns...")
    df_result = df[['user', 'location_id', 'time']]
    
    print(f"Saving to {output_file}...")
    df_result.to_csv(output_file, index=False)
    print("Conversion complete.")

if __name__ == "__main__":
    convert()
