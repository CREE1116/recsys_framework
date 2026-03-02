import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from src.data_loader import DataLoader
from src.models.general.slim import SLIM

def test_deduplication():
    print("\n--- Testing Deduplication ---")
    data = {
        'user_id': [0, 0, 0, 1, 1],
        'item_id': [10, 10, 20, 30, 30],
        'timestamp': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    config = {
        'dataset_name': 'test',
        'data_path': 'dummy',
        'separator': '\t',
        'columns': ['user_id', 'item_id', 'timestamp'],
        'min_user_interactions': 0,
        'min_item_interactions': 0,
        'split_method': 'random',
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'evaluation': {'validation_method': 'full'}
    }
    
    # Mocking _load_data and _filter_interactions since we want to test _process_data logic
    class MockDataLoader(DataLoader):
        def _load_data(self): return df
        def _filter_interactions(self, df): return df
        def _save_to_cache(self): pass
    
    dl = MockDataLoader(config)
    print(f"Original interactions: {len(data['user_id'])}")
    print(f"Deduplicated interactions: {len(dl.df)}")
    assert len(dl.df) == 3, f"Expected 3 unique interactions, got {len(dl.df)}"
    print("Deduplication test PASSED")

def test_slim_parallel():
    print("\n--- Testing SLIM Parallelization ---")
    n_users, n_items = 100, 50
    rows = np.random.randint(0, n_users, 500)
    cols = np.random.randint(0, n_items, 500)
    data = np.ones(500)
    
    train_df = pd.DataFrame({'user_id': rows, 'item_id': cols})
    
    class MockDL:
        def __init__(self):
            self.n_users = n_users
            self.n_items = n_items
            self.train_df = train_df

    dl = MockDL()
    config = {
        'model': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 10, 'n_jobs': -1},
        'device': 'cpu'
    }
    
    slim = SLIM(config, dl)
    
    start_time = time.time()
    slim.fit(dl)
    end_time = time.time()
    
    print(f"SLIM Training took {end_time - start_time:.2f} seconds")
    print(f"W matrix shape: {slim.W.shape}")
    assert slim.W.shape == (n_items, n_items), "W matrix shape mismatch"
    print("SLIM Parallelization test PASSED")

def test_vectorized_split():
    print("\n--- Testing Vectorized Split Consistency ---")
    n_users = 100
    n_items_per_user = 20
    data = {
        'user_id': np.repeat(np.arange(n_users), n_items_per_user),
        'item_id': np.tile(np.arange(n_items_per_user), n_users),
        'timestamp': np.tile(np.arange(n_items_per_user), n_users)
    }
    df = pd.DataFrame(data)
    
    config = {
        'dataset_name': 'test_split',
        'data_path': 'dummy',
        'separator': '\t',
        'columns': ['user_id', 'item_id', 'timestamp'],
        'min_user_interactions': 0,
        'min_item_interactions': 0,
        'split_method': 'temporal_ratio',
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'evaluation': {'validation_method': 'full'},
        'device': 'cpu'
    }
    
    class MockDL(DataLoader):
        def _load_data(self): return df
        def _filter_interactions(self, df): return df
        def _save_to_cache(self): pass

    dl = MockDL(config)
    
    # Expected: 20 * 0.8 = 16 train, 20 * 0.1 = 2 valid, remainder 2 test per user
    print(f"Train size: {len(dl.train_df)}, Valid size: {len(dl.valid_df)}, Test size: {len(dl.test_df)}")
    assert len(dl.train_df) == n_users * 16
    assert len(dl.valid_df) == n_users * 2
    assert len(dl.test_df) == n_users * 2
    
    # Check for leakage
    train_ids = dl.train_df.groupby('user_id')['item_id'].apply(set)
    test_ids = dl.test_df.groupby('user_id')['item_id'].apply(set)
    for u in range(n_users):
        assert not (train_ids[u] & test_ids[u]), f"Leakage detected for user {u}"
        
    print("Vectorized split test PASSED")

if __name__ == "__main__":
    test_deduplication()
    test_slim_parallel()
    test_vectorized_split()
