import torch
import scipy.sparse as sp
import numpy as np

from src.models.csar.ASPIRE import ASPIRE

class DummyLoader:
    def __init__(self, u, i):
        self.n_users = u
        self.n_items = i
        import pandas as pd
        self.train_df = pd.DataFrame({'user_id': np.random.randint(0, u, 1000), 'item_id': np.random.randint(0, i, 1000)})

config = {
    'model': {'name': 'spectral_tikhonov_lira', 'alpha': 20.0, 'beta': 0.8, 'target_energy': 0.99, 'visualize': True},
    'dataset_name': 'dummy_dataset',
    'device': 'cpu'
}

print("Testing ASPIRE sigma vs h(sigma) visualization...")
loader = DummyLoader(100, 100)
model = ASPIRE(config, loader)
model.fit(loader)
print("Success!")
