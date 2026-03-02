import torch
import scipy.sparse as sp
import numpy as np
import time

from src.models.csar.LIRALayer import ChebyshevLIRALayer

print("Generating mock sparse data...")
num_users = 1000
num_items = 1000
X_np = np.random.binomial(1, 0.01, size=(num_users, num_items)).astype(np.float32)
X_sparse = sp.csr_matrix(X_np)

print("Testing FilterLIRA...")
layer = ChebyshevLIRALayer(reg_lambda=500, K=3, threshold=1e-5)
layer.build(X_sparse)

print("Success!")
