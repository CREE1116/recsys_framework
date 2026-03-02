import torch
import numpy as np
import scipy.sparse as sp

from src.models.csar.AsymmetricLIRA import AsymmetricLIRA

class DummyLoader:
    def __init__(self, u, i):
        self.n_users = u
        self.n_items = i
        import pandas as pd
        self.train_df = pd.DataFrame({'user_id': [0, 1, 2], 'item_id': [0, 1, 2]})

config = {
    'model': {'reg_lambda': 500.0},
    'dataset_name': 'dummy',
    'device': 'cpu'
}

print("Testing Asymmetric LIRA...")
loader = DummyLoader(100, 100)
model = AsymmetricLIRA(config, loader)

users = torch.tensor([0, 1, 2])
res = model(users)
print(f"Pred shape: {res.shape}")
print("Success!")
