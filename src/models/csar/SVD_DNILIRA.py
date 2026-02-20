import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from src.models.base_model import BaseModel
from src.utils.svd_manager import SVDCacheManager

class SVD_DNILIRA(BaseModel):
    """
    Memory-efficient SVD-based DNILIRA with Analysis Visualizations
    
    1. Symmetric: SVD -> Latent Wiener Filter
    2. Asymmetric: Sparse Transition Flow
    3. Visualizations: Spectrum, Flow heatmap, Final Kernel heatmap
    """
    def __init__(self, config, data_loader):
        super(SVD_DNILIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_cfg = config.get('model', {})
        self.embedding_dim = model_cfg.get('embedding_dim', 256)
        self.reg_lambda    = model_cfg.get('reg_lambda', 500.0)
        self.beta          = model_cfg.get('beta', 0.1)
        self.eps           = float(model_cfg.get('eps', 1e-8))
        self.visualize     = model_cfg.get('visualize', True)

        # 1. Build Data structures
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        sequences = self._extract_sequences(data_loader)
        device = self.device
        
        # 2. Symmetric Part (SVD-based LIRA)
        print(f"[SVD_DNILIRA] Computing/Loading SVD (k={self.embedding_dim}) ...")
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(self.train_matrix_csr, k=self.embedding_dim, 
                                                dataset_name=config.get('dataset_name', 'unknown'))
        
        self.singular_values = s.to(device)
        self.V_k = v.to(device) # [embedding_dim, n_items] -> wait, manager returns [n_items, k]?
        # Standardize vt shape: manager returns vt as [k, n_items]
        if self.V_k.shape[0] != self.n_items:
            # vt case
            self.V_k = self.V_k.t() # [n_items, k]
            
        s2 = self.singular_values.pow(2)
        lira_f = s2 / (s2 + self.reg_lambda)
        
        # Store Low-rank components
        # S = (V * f) @ V.T
        self.V_weighted = self.V_k * lira_f.view(1, -1)
        
        # Exact Normalization for S based on Energy (Absolute sum)
        # S_ij = sum_k V_ik * f_k * V_jk
        # d_s_i = sum_j |sum_k V_ik * f_k * V_jk|
        # For memory efficiency, we compute d_s in batches if N is large.
        print("[SVD_DNILIRA] Computing Symmetric Normalization Factors ...")
        M = self.n_items
        d_s = torch.zeros(M, device=device)
        batch_size = 2000
        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            # [batch, k] @ [k, N] -> [batch, N]
            S_batch = torch.mm(self.V_weighted[i:end], self.V_k.t())
            d_s[i:end] = S_batch.abs().sum(dim=1)
            del S_batch
        
        self.register_buffer('d_s_inv_sqrt', torch.pow(d_s + self.eps, -0.5))

        # 3. Asymmetric Part (Sparse Transition Flow)
        print("[SVD_DNILIRA] Building Sparse Transition Flow ...")
        row, col, val = [], [], []
        counts_dict = {}
        out_degrees = np.zeros(self.n_items)
        in_degrees = np.zeros(self.n_items)
        
        for seq in sequences:
            for t in range(len(seq) - 1):
                i, j = seq[t], seq[t + 1]
                counts_dict[(i, j)] = counts_dict.get((i, j), 0) + 1.0
                out_degrees[i] += 1
                in_degrees[j] += 1
        
        for (i, j), count in counts_dict.items():
            t_ji = counts_dict.get((j, i), 0.0)
            a_ij = count - t_ji
            if a_ij != 0:
                row.append(i)
                col.append(j)
                val.append(a_ij)
        
        A_sparse_coo = csr_matrix((val, (row, col)), shape=(self.n_items, self.n_items))
        self.A_sparse = self._to_torch_sparse(A_sparse_coo).to(device)
        
        # d_a_inv_sqrt = In-degree + Out-degree
        d_a = torch.from_numpy((out_degrees + in_degrees).astype(np.float32)).to(device)
        self.register_buffer('d_a_inv_sqrt', torch.pow(d_a + self.eps, -0.5))
        
        # Optimized MPS fallback: Keep CPU copies for sparse mm
        if device.type == 'mps':
            self.A_sparse_cpu = self.A_sparse.to('cpu')
            self.d_a_inv_sqrt_cpu = self.d_a_inv_sqrt.to('cpu')
        else:
            self.A_sparse_cpu = None
            self.d_a_inv_sqrt_cpu = None

        # 4. Visualization & Analysis
        if self.visualize:
            self._perform_analysis(config)

    def _perform_analysis(self, config):
        try:
            analysis_dir = SVDCacheManager.get_analysis_dir(config)
            os.makedirs(analysis_dir, exist_ok=True)
            print(f"[SVD_DNILIRA] Saving analysis to {analysis_dir} ...")
            
            # --- 1. SVD Energy Analysis ---
            s_vals = self.singular_values.cpu().numpy()
            s2 = s_vals**2
            total_s2 = s2.sum() # This is only for the top-k! 
            # Note: SVDCacheManager doesn't return total_energy of ALL singular values easily if we only compute k.
            # But we can show energy within k.
            cum_energy = np.cumsum(s2) / np.sum(s2)
            
            analysis_stats = {
                "svd": {
                    "k": self.embedding_dim,
                    "singular_values_sum": float(s_vals.sum()),
                    "energy_top_k": float(np.sum(s2)),
                    "energy_distribution": [float(cum_energy[i]) for i in range(0, len(cum_energy), max(1, len(cum_energy)//10))]
                }
            }

            # --- 2. Matrix Statistics (Sampled 1000x1000) ---
            sample_size = min(1000, self.n_items)
            
            # Symmetric Kernel
            device = self.device
            S_sample = torch.mm(self.V_weighted[:sample_size], self.V_k[:sample_size].t())
            S_sample = self.d_s_inv_sqrt[:sample_size].view(-1, 1) * S_sample * self.d_s_inv_sqrt[:sample_size].view(1, -1)
            S_np = S_sample.cpu().numpy()
            
            analysis_stats["symmetric_kernel_sampled"] = {
                "mean": float(S_np.mean()),
                "std": float(S_np.std()),
                "min": float(S_np.min()),
                "max": float(S_np.max()),
                "sparsity": float(np.mean(S_np == 0))
            }

            # Asymmetric Flow
            try:
                if self.n_items <= 10000:
                    A_dense_sample = self.A_sparse.to_dense()[:sample_size, :sample_size]
                    A_sample = self.d_a_inv_sqrt[:sample_size].view(-1, 1) * A_dense_sample * self.d_a_inv_sqrt[:sample_size].view(1, -1)
                    A_np = A_sample.cpu().numpy()
                else:
                    A_np = np.zeros((sample_size, sample_size))
                
                analysis_stats["asymmetric_flow_sampled"] = {
                    "mean": float(A_np.mean()),
                    "std": float(A_np.std()),
                    "min": float(A_np.min()),
                    "max": float(A_np.max()),
                    "sparsity": float(np.mean(A_np == 0))
                }
            except:
                A_np = np.zeros((sample_size, sample_size))

            # Unified Kernel
            K_np = S_np + self.beta * A_np
            analysis_stats["unified_kernel_sampled"] = {
                "beta": self.beta,
                "mean": float(K_np.mean()),
                "std": float(K_np.std()),
                "min": float(K_np.min()),
                "max": float(K_np.max())
            }

            # Save stats to JSON
            with open(os.path.join(analysis_dir, 'analysis_results.json'), 'w') as f:
                json.dump(analysis_stats, f, indent=4)

            # --- 3. Visualization Plots ---
            # Spectrum Plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(s_vals)
            plt.yscale('log')
            plt.title("SVD Spectrum")
            
            plt.subplot(1, 3, 2)
            plt.plot(cum_energy)
            plt.title(f"Cumulative Energy (Top-k)")
            
            filter_w = (s2 / (s2 + self.reg_lambda))
            plt.subplot(1, 3, 3)
            plt.plot(filter_w)
            plt.title(fr"Filtering ($\lambda={self.reg_lambda}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'svd_spectrum.png'))
            plt.close()
            
            # Symmetric Kernel Heatmap
            plt.figure(figsize=(8, 6))
            vmax = np.percentile(np.abs(S_np), 99)
            sns.heatmap(S_np, cmap='viridis', vmax=vmax, vmin=-vmax/10)
            plt.title(f"Symmetric Kernel (Subset {sample_size})")
            plt.savefig(os.path.join(analysis_dir, 'viz_S_norm.png'))
            plt.close()
            
            # Asymmetric Flow Heatmap
            if self.n_items <= 10000:
                plt.figure(figsize=(8, 6))
                vmax_a = np.percentile(np.abs(A_np), 99)
                sns.heatmap(A_np, cmap='RdBu_r', center=0, vmax=vmax_a, vmin=-vmax_a)
                plt.title(f"Asymmetric Flow (Subset {sample_size})")
                plt.savefig(os.path.join(analysis_dir, 'viz_A_norm.png'))
                plt.close()
            
            # Final Kernel Heatmap
            plt.figure(figsize=(8, 6))
            vmax_k = np.percentile(np.abs(K_np), 99)
            sns.heatmap(K_np, cmap='viridis', vmax=vmax_k, vmin=-vmax_k/10)
            plt.title(f"Unified Kernel (S + {self.beta}*A)")
            plt.savefig(os.path.join(analysis_dir, 'viz_K_unified.png'))
            plt.close()
            
            print(f"[SVD_DNILIRA] Analysis completed.")
        except Exception as e:
            print(f"[SVD_DNILIRA] Visualization/Stats failed: {e}")
            import traceback
            traceback.print_exc()

    def _to_torch_sparse(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data.astype(np.float32))
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

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

    def forward(self, users, mask_observed=True):
        device = self.device
        batch_users_np = users.cpu().numpy()
        X_u_sparse = self.train_matrix_csr[batch_users_np]
        X_u = torch.from_numpy(X_u_sparse.toarray()).float().to(device)
        
        # 1. Symmetric Prediction
        X_s = X_u * self.d_s_inv_sqrt
        latent_s = torch.mm(X_s, self.V_weighted)
        scores_s = torch.mm(latent_s, (self.V_k * self.d_s_inv_sqrt.view(-1, 1)).t())
        
        # 2. Asymmetric Prediction
        if self.A_sparse.is_sparse and device.type == 'mps':
            # Case 1: MPS Device (Sparse MM fallback to CPU)
            X_a_cpu = (torch.from_numpy(X_u_sparse.toarray()).float()) * self.d_a_inv_sqrt_cpu
            # Originally was X_a * A.t(). We want X_a * A.
            # (A.t() @ X_a.t()).t() = X_a @ A. Correct.
            res_a = torch.sparse.mm(self.A_sparse_cpu.t(), X_a_cpu.t()).t()
            scores_a = (res_a * self.d_a_inv_sqrt_cpu).to(device)
        else:
            # Case 2: CPU or CUDA (Normal Sparse MM)
            X_a = X_u * self.d_a_inv_sqrt
            # (A.t() @ X_a.t()).t() = X_a @ A
            scores_a = torch.sparse.mm(self.A_sparse.t(), X_a.t()).t()
            scores_a = scores_a * self.d_a_inv_sqrt
        
        scores = scores_s + self.beta * scores_a
        
        if mask_observed:
            rows, cols = X_u.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
            
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # Optimized for pair prediction
        device = self.device
        scores = []
        batch_size = 512
        for i in range(0, len(user_ids), batch_size):
            u_batch = user_ids[i:i+batch_size]
            i_batch = item_ids[i:i+batch_size]
            full_scores = self.forward(u_batch, mask_observed=False)
            scores.append(full_scores.gather(1, i_batch.view(-1, 1)).squeeze())
        return torch.cat(scores)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def get_final_item_embeddings(self):
        return self.V_weighted
