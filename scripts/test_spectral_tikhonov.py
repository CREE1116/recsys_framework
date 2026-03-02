import torch
import scipy.sparse as sp
import numpy as np

from src.models.csar.SpectralTikhonovLIRA import SpectralTikhonovLIRA

class DummyLoader:
    def __init__(self, u, i):
        self.n_users = u
        self.n_items = i
        import pandas as pd
        self.train_df = pd.DataFrame({'user_id': np.random.randint(0, u, 1000), 'item_id': np.random.randint(0, i, 1000)})

config = {
    'model': {'alpha': 500.0, 'beta': 1.0, 'target_energy': 0.99},
    'dataset_name': 'dummy',
    'device': 'cpu'
}

print("Testing SpectralTikhonovLIRA target_energy...")
loader = DummyLoader(100, 100)
model = SpectralTikhonovLIRA(config, loader)

users = torch.tensor([0, 1, 2])
res = model(users)
print(f"Pred shape: {res.shape}")
print("Success!")
