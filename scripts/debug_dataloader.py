import pandas as pd
import numpy as np
from src.data_loader import DataLoader
import os

def debug_dataloader():
    print("--- Debugging DataLoader ---")
    data = {
        'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
        'item_id': ['i1', 'i2', 'i3', 'i1', 'i2', 'i4'],
        'timestamp': [1, 2, 3, 1, 2, 4],
        'rating': [5, 4, 3, 5, 4, 2]
    }
    df = pd.DataFrame(data)
    data_path = 'debug_data.csv'
    df.to_csv(data_path, index=False, sep='\t')
    
    config = {
        'dataset_name': 'debug',
        'data_path': data_path,
        'separator': '\t',
        'columns': ['user_id', 'item_id', 'timestamp', 'rating'],
        'min_user_interactions': 0,
        'min_item_interactions': 0,
        'split_method': 'loo',
        'evaluation': {'validation_method': 'uni99'},
        'device': 'cpu',
        'has_header': True
    }
    
    try:
        dl = DataLoader(config)
        print("DataLoader initialized successfully.")
        print(f"Train size: {len(dl.train_df)}")
        print(f"Valid size: {len(dl.valid_df)}")
        print(f"Test size: {len(dl.test_df)}")
    except Exception as e:
        import traceback
        print(f"FAILED with error: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists('cache/debug_loo_uni99.pkl'):
            os.remove('cache/debug_loo_uni99.pkl')

if __name__ == "__main__":
    debug_dataloader()
