import torch
import torch.nn as nn
from src.models.base_model import BaseModel
import numpy as np

class SymmetricMSLIRA(BaseModel):
    """
    Symmetric Multi-scale LIRA (SMS-LIRA)
    
    Breakthrough: 
    Combines Wiener-filtered local terrain with spectral-diffused global manifold.
    Ensures numerical stability via full symmetry.
    """
    def __init__(self, config, data_loader):
        super(SymmetricMSLIRA, self).__init__(config, data_loader)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        model_cfg = config.get('model', {})
        self.reg_lambda = model_cfg.get('reg_lambda', 4673.13)
        self.lambda_d = model_cfg.get('lambda_d', 0.6)
        self.beta = model_cfg.get('beta', 0.5)
        self.k = model_cfg.get('k', 2)
        self.eps = 1e-8

        # 1. Base Terrain (S: Wiener Filter)
        # Using a dense approach as interaction matrices for ML are manageable (~1.5GB for ml-1m)
        print(f"[SMS-LIRA] Initializing on {self.device}...")
        X = torch.from_numpy(data_loader.train_matrix.toarray()).float().to(self.device)
        n_items = X.size(1)
        G = torch.mm(X.t(), X)
        I = torch.eye(n_items, device=self.device)
        
        print(f"[SMS-LIRA] Building Local Terrain (lambda={self.reg_lambda})...")
        # S = G (G + lambda I)^-1
        self.S = torch.mm(G, torch.linalg.inv(G + self.reg_lambda * I))

        # 2. Spectral Diffusion (Phi: Higher-order Proximity)
        # Normalized Adjacency A_hat = D^-1/2 G D^-1/2
        d_inv_sqrt = torch.pow(G.sum(dim=1) + self.eps, -0.5)
        A_hat = d_inv_sqrt.view(-1, 1) * G * d_inv_sqrt.view(1, -1)
        
        # Diffusion Operator P = (1 - lambda_d)I + lambda_d * A_hat
        P = (1 - self.lambda_d) * I + self.lambda_d * A_hat
        
        print(f"[SMS-LIRA] Diffusing Global Manifold (k={self.k}, lambda_d={self.lambda_d})...")
        P_k = torch.matrix_power(P, self.k)
        
        # Phi = P_k @ G @ P_k (Symmetric Multi-scale interaction)
        # This encapsulates k-hop symmetric connections weighted by the base manifold G
        self.Phi = torch.mm(torch.mm(P_k, G), P_k)

        # 3. Final Fusion & Dual Normalization
        # K = S (Local) + beta * Phi (Global)
        K_raw = self.S + self.beta * self.Phi
        
        # Symmetrical Normalization for Rank Stability
        # We use absolute sums to handle potential negative values in K_raw (though G is PSD)
        d_norm = torch.pow(K_raw.abs().sum(dim=1) + self.eps, -0.5)
        self.K_final = d_norm.view(-1, 1) * K_raw * d_norm.view(1, -1)
        
        self.register_buffer('kernel', self.K_final)
        self.train_matrix_csr = data_loader.train_matrix

    def forward(self, users, mask_observed=True):
        batch_users = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users].toarray()).float().to(self.device)
        
        # Prediction: R = X @ K
        scores = torch.mm(X_u, self.kernel)
        
        if mask_observed:
            # Mask out items the user has already interacted with
            scores[X_u > 0] = -1e9
            
        return scores
