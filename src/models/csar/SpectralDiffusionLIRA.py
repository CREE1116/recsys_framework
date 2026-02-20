import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.svd_manager import SVDCacheManager

class SpectralDiffusionLIRA(BaseModel):
    """
    Spectral Diffusion LIRA (SpectralDiffusionLIRA)
    
    Captures directional flow through a Graph Diffusion Filter:
    P = (1 - lambda_d)I + lambda_d * A_hat
    P_k = P^k
    """
    def __init__(self, config, data_loader):
        super(SpectralDiffusionLIRA, self).__init__(config, data_loader)
        self.n_items = data_loader.n_items
        self.n_users = data_loader.n_users
        
        model_cfg = config.get('model', {})
        self.reg_lambda = model_cfg.get('reg_lambda', 500.0)
        self.beta = model_cfg.get('beta', 0.5)
        self.lambda_d = model_cfg.get('lambda_d', 0.1)
        self.k = int(model_cfg.get('k', 2))
        self.eps = float(model_cfg.get('eps', 1e-8))
        self.visualize = model_cfg.get('visualize', True)

        device = self.device

        # 1. Base Terrain (Wiener Filter S)
        print("[SpectralDiffusion] Preparing base terrain (Wiener Filter)...")
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = torch.from_numpy(self.train_matrix_csr.toarray()).float().to(device)
        G = torch.mm(X.t(), X)
        I = torch.eye(self.n_items, device=device)
        S = torch.mm(G, torch.linalg.inv(G + self.reg_lambda * I))

        # 2. Diffusion Operator (P) 구축
        print("[SpectralDiffusion] Building Diffusion Operator P...")
        D_inv_sqrt = torch.pow(G.sum(dim=1) + self.eps, -0.5)
        A_hat = D_inv_sqrt.view(-1, 1) * G * D_inv_sqrt.view(1, -1)
        P = (1 - self.lambda_d) * I + self.lambda_d * A_hat

        # 3. Interest Manifold Shaping (P^k)
        print(f"[SpectralDiffusion] Shaping Interest Manifold (k={self.k})...")
        P_k = torch.matrix_power(P, self.k)

        # 4. Directional Diffusion (Phi_latent)
        print("[SpectralDiffusion] Extracting Directional Momentum...")
        sequences = self._extract_sequences(data_loader)
        T_counts = self._build_transition_matrix(sequences)
        A_raw = T_counts - T_counts.t()
        Phi_latent = torch.mm(torch.mm(P_k, A_raw), P_k)

        # 5. Fusion & Dual-Norm
        K_unified = S + self.beta * Phi_latent
        d = K_unified.abs().sum(dim=1).clamp(min=self.eps).pow(-0.5)
        self.K_final = d.view(-1, 1) * K_unified * d.view(1, -1)
        
        self.register_buffer('K_buffer', self.K_final)

        # Analysis & Visualization
        if self.visualize:
            self._perform_analysis(config, S, P_k, A_raw, Phi_latent, self.K_final)

    def _perform_analysis(self, config, S, P_k, A_raw, Phi_latent, K_final):
        try:
            analysis_dir = SVDCacheManager.get_analysis_dir(config)
            os.makedirs(analysis_dir, exist_ok=True)
            print(f"[SpectralDiffusion] Saving diagnostic analysis to {analysis_dir} ...")
            
            sample_size = min(1000, self.n_items)
            
            def get_stats(tensor, name):
                t_np = tensor[:sample_size, :sample_size].detach().cpu().numpy()
                return {
                    "mean": float(t_np.mean()),
                    "std": float(t_np.std()),
                    "min": float(t_np.min()),
                    "max": float(t_np.max()),
                    "sparsity": float(np.mean(np.abs(t_np) < 1e-6)),
                    "spectral_radius_approx": float(torch.norm(tensor, p=2).item()) if tensor.is_floating_point() else 0.0
                }

            analysis_stats = {
                "Symmetric_Skeleton_S": get_stats(S, "S"),
                "Diffusion_Kernel_Pk": get_stats(P_k, "Pk"),
                "Raw_Asymmetric_Flow_A": get_stats(A_raw, "A"),
                "Latent_Flow_Phi": get_stats(Phi_latent, "Phi"),
                "Final_Kernel_K": get_stats(K_final, "K"),
                "Hyperparameters": {
                    "reg_lambda": self.reg_lambda,
                    "beta": self.beta,
                    "lambda_d": self.lambda_d,
                    "k": self.k
                }
            }

            with open(os.path.join(analysis_dir, 'spectral_diffusion_stats.json'), 'w') as f:
                json.dump(analysis_stats, f, indent=4)

            # Heatmaps
            matrices = [
                (S, "S_Skeleton", "viridis"),
                (P_k, "P_k_Diffusion", "magma"),
                (Phi_latent, "Phi_Latent_Flow", "RdBu_r"),
                (K_final, "K_Final_Unified", "viridis")
            ]

            for mat, name, cmap in matrices:
                plt.figure(figsize=(8, 6))
                m_np = mat[:sample_size, :sample_size].detach().cpu().numpy()
                vmax = np.percentile(np.abs(m_np), 99)
                center = 0 if cmap == "RdBu_r" else None
                sns.heatmap(m_np, cmap=cmap, center=center, vmax=vmax, vmin=-vmax if center is not None else -vmax/10)
                plt.title(f"{name} (Sample {sample_size}x{sample_size})")
                plt.savefig(os.path.join(analysis_dir, f'viz_{name}.png'))
                plt.close()

            print(f"[SpectralDiffusion] Analysis completed.")
        except Exception as e:
            print(f"[SpectralDiffusion] Analysis failed: {e}")

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def _extract_sequences(self, data_loader):
        train_df = data_loader.train_df.copy()
        if 'timestamp' not in train_df.columns:
            train_df['timestamp'] = train_df.index
        train_df = train_df.sort_values(by=['user_id', 'timestamp'])
        return train_df.groupby('user_id')['item_id'].apply(list).tolist()

    def _build_transition_matrix(self, sequences):
        n = self.n_items
        T = torch.zeros(n, n, device=self.device)
        for seq in sequences:
            if len(seq) < 2: continue
            for i in range(len(seq) - 1):
                T[seq[i], seq[i+1]] += 1.0
        return T

    def forward(self, users, mask_observed=True):
        device = self.device
        batch_users_np = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users_np].toarray()).float().to(device)
        scores = torch.mm(X_u, self.K_buffer)
        if mask_observed:
            rows, cols = X_u.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
        return scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def predict_for_pairs(self, user_ids, item_ids):
        device = self.device
        batch_users_np = user_ids.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users_np].toarray()).float().to(device)
        relevant_K = self.K_buffer[:, item_ids]
        scores = (X_u * relevant_K.t()).sum(dim=1)
        return scores

    def get_final_item_embeddings(self):
        return self.K_buffer
