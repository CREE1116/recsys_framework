import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy.sparse.linalg import svds
from src.utils.svd_manager import SVDCacheManager

class LIRALayer(nn.Module):
    """
    LIRA - Linear Item Representation Analysis via Dual Ridge Regression
    """
    def __init__(self, reg_lambda=500.0, normalize=True):
        super(LIRALayer, self).__init__()
        self.reg_lambda = reg_lambda
        self.normalize = normalize

        self.register_buffer('S', torch.empty(0))           # full-rank

    @torch.no_grad()
    def build(self, X_sparse):
        """
        Dual Ridge Regression (VCV mode)
        S = X.T @ (X @ X.T + lambda * I)^(-1) @ X
        Here, V = X.T (Items as vectors in User space)
        C = (X @ X.T + lambda * I)^(-1) (User-User correlation inverse)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'

        print(f"[LIRA] Building with λ={self.reg_lambda} ..")

        # 1. Convert to dense for Kernel computation (N x M)
        X = torch.from_numpy(X_sparse.toarray()).float().to(device)
        n_users, n_items = X.shape

        # 2. Compute Gram Matrix K = X @ X.T (N x N)
        K = torch.mm(X, X.t())
        
        # Store Gram Matrix for visualization
        self.gram_matrix = K
        
        # 3. Regularize: K_reg = K + lambda * I
        K_reg = K + self.reg_lambda * torch.eye(n_users, device=device)

        # 4. Compute C = inverse(K_reg)
        C = torch.linalg.inv(K_reg)

        # 5. Compute S = X.T @ C @ X
        S = torch.mm(torch.mm(X.t(), C), X)
        
        if self.normalize:
            self.S_raw = S.clone()
            d = S.abs().sum(dim=1)
            d[d == 0] = 1.0
            d_inv_sqrt = d.pow(-0.5)
            S = d_inv_sqrt.unsqueeze(1) * S * d_inv_sqrt.unsqueeze(0)
            print("[LIRA] Applied Symmetric Normalization (D^-0.5 S D^-0.5)")
        else:
             self.S_raw = S.clone()

        self.S = S
        self.C = C
        print(f"[LIRA] S built. Norm: {S.norm():.2f}")

    def visualize_matrices(self, X_sparse=None, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        S_tensor = self.S.detach().cpu()
        S_raw_tensor = self.S_raw.detach().cpu() if hasattr(self, 'S_raw') else S_tensor
        gram_tensor = self.gram_matrix.detach().cpu() if hasattr(self, 'gram_matrix') else self.C.detach().cpu()
        
        S_np = S_tensor.numpy()
        S_raw_np = S_raw_tensor.numpy()
        K_np = gram_tensor.numpy()

        metrics = {}
        
        def get_stats(name, tensor):
            return {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "sparsity": float((tensor.abs() < 1e-6).sum() / tensor.numel())
            }

        metrics['S_raw'] = get_stats('S_raw', S_raw_tensor)
        metrics['S_norm'] = get_stats('S_norm', S_tensor)
        metrics['K_Gram'] = get_stats('K_Gram', gram_tensor)
        
        if save_dir:
            metrics_path = os.path.join(save_dir, 'lira_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

        def plot_heatmap(matrix, name, filename, cmap='viridis', center=None, vmin=None, vmax=None):
            plt.figure(figsize=(10, 8))
            N, M = matrix.shape
            if N > 1000 or M > 1000:
                step_n = max(1, N // 1000)
                step_m = max(1, M // 1000)
                matrix_viz = matrix[::step_n, ::step_m]
            else:
                matrix_viz = matrix
            sns.heatmap(matrix_viz, cmap=cmap, center=center, square=True, vmax=vmax, vmin=vmin)
            plt.title(f'{name}')
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, filename), dpi=150)
            plt.close()

        plot_heatmap(K_np, 'Gram Matrix K', 'viz_K_Gram.png')
        vmax = np.percentile(np.abs(S_raw_np), 99)
        plot_heatmap(S_raw_np, 'S Matrix (Raw)', 'viz_S_Raw.png', vmax=vmax, vmin=-vmax/10)
        vmax_norm = np.percentile(np.abs(S_np), 99)
        plot_heatmap(S_np, 'S Matrix (Norm)', 'viz_S_Norm.png', vmax=vmax_norm, vmin=-vmax_norm/10)

    @torch.no_grad()
    def forward(self, X_batch, mask_observed=True):
        if self.S.numel() == 0:
            raise RuntimeError("build() must be called first")

        # 2. Score Computation: O(N * M)
        scores = torch.mm(X_batch, self.S)
        
        # 4. Optimized Masking
        if mask_observed:
            rows, cols = X_batch.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
            
        return scores


class LightLIRALayer(nn.Module):
    """
    LightLIRA - Scalable Low-rank LIRA (SVD-based)
    Optimized for memory efficiency via batch-wise processing.
    """
    def __init__(self, k=200, reg_lambda=500.0, normalize=True):
        super(LightLIRALayer, self).__init__()
        self.k = k
        self.reg_lambda = reg_lambda
        self.normalize = normalize

        self.register_buffer('V_k', torch.empty(0))           
        self.register_buffer('U_k', torch.empty(0))           
        self.register_buffer('filter_diag', torch.empty(0))    
        self.register_buffer('S_diag_inv', torch.empty(0))     
        self.register_buffer('singular_values', torch.empty(0)) 

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[LightLIRA] Building Model with k={self.k}, λ={self.reg_lambda} (Cache: {dataset_name}) ..")

        # Use SVDCacheManager for simplified SVD handling
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        
        # Proper buffer update via assignment (names are already registered)
        self.U_k = u.to(device)
        self.singular_values = s.to(device)
        self.V_k = v.to(device)
        
        s2 = self.singular_values.pow(2)
        self.filter_diag = s2 / (s2 + self.reg_lambda)
        
        # [Optimization 1] 배치 단위 정규화 계수 계산
        if self.normalize:
            print("[LightLIRA] Computing Normalization degrees (Batch-wise)...")
            M = self.V_k.size(0)
            d = torch.zeros(M, device=device)
            batch_size = 1000
            
            for i in range(0, M, batch_size):
                end = min(i + batch_size, M)
                mat_batch = (self.V_k[i:end] * self.filter_diag) @ self.V_k.t()
                d[i:end] = mat_batch.abs().sum(dim=1)
                del mat_batch
            
            d[d == 0] = 1.0
            self.register_buffer('S_diag_inv', d.pow(-0.5))
            print("[LightLIRA] Applied Symmetric Normalization.")
        else:
            # Prevent RuntimeError in forward if normalization is checked
            # We initialize with a safe dummy or ensure size-0 isn't used
            self.register_buffer('S_diag_inv', torch.ones(self.V_k.size(0), device=device))

    @torch.no_grad()
    def forward(self, X_batch, mask_observed=True):
        if self.normalize:
             X_batch = X_batch * self.S_diag_inv.unsqueeze(0)
        
        # Efficient Inference: (X @ V) * f @ V.T
        latent_rep = (torch.mm(X_batch, self.V_k)) * self.filter_diag
        scores = torch.mm(latent_rep, self.V_k.t())
        
        if self.normalize:
            scores = scores * self.S_diag_inv.unsqueeze(0)
            
        if mask_observed:
            # [Optimization] 전체 스캔 대신 관측된(non-zero) 인덱스만 추출
            rows, cols = X_batch.nonzero(as_tuple=True)
            # 해당 위치만 -1e9로 직접 타격 (In-place 연산)
            scores[rows, cols] = -1e9
            
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        """[Optimization 2] 메모리 점유를 최소화하는 분석 루틴"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import json

        if save_dir: os.makedirs(save_dir, exist_ok=True)
        M, N = self.V_k.size(0), self.U_k.size(0)
        metrics = {}
        
        # --- (Stats Calculation remains same) ---
        def get_batch_stats(is_C=False, is_S_raw=False):
            """전체 행렬을 만들지 않고 통계량 계산"""
            size = N if is_C else M
            total_sum, total_sq_sum = 0.0, 0.0
            total_count = size * size
            zero_count = 0
            max_val, min_val = -float('inf'), float('inf')
            
            batch_size = 500
            for i in range(0, size, batch_size):
                end = min(i + batch_size, size)
                if is_C:
                    inv_vals = 1.0 / (self.singular_values.pow(2) + self.reg_lambda)
                    mat_batch = (self.U_k[i:end] * inv_vals.unsqueeze(0)) @ self.U_k.t()
                elif is_S_raw:
                    mat_batch = (self.V_k[i:end] * self.filter_diag) @ self.V_k.t()
                else: # S_norm
                    mat_batch = (self.V_k[i:end] * self.filter_diag) @ self.V_k.t()
                    if self.normalize:
                        mat_batch = self.S_diag_inv[i:end].unsqueeze(1) * mat_batch * self.S_diag_inv.unsqueeze(0)

                max_val = max(max_val, mat_batch.max().item())
                min_val = min(min_val, mat_batch.min().item())
                total_sum += mat_batch.sum().item()
                total_sq_sum += mat_batch.pow(2).sum().item()
                zero_count += (mat_batch.abs() < 1e-6).sum().item()
                del mat_batch

            mean = total_sum / total_count
            std = (total_sq_sum / total_count - mean**2)**0.5
            return {"min": min_val, "max": max_val, "mean": mean, "std": std, "sparsity": zero_count/total_count}

        print(f"[LightLIRA] Calculating metrics (Lightweight={lightweight})...")
        metrics['S_raw'] = get_batch_stats(is_S_raw=True)
        metrics['S_norm'] = get_batch_stats()
        metrics['C_approx'] = get_batch_stats(is_C=True)
        
        # Energy & Spectrum stats
        s_vals = self.singular_values.detach().cpu().numpy()
        s2 = s_vals**2
        cum_energy = np.cumsum(s2)
        
        metrics['SingularValues'] = {"mean": float(s_vals.mean()), "std": float(s_vals.std())}
        filter_w = self.filter_diag.detach().cpu().numpy()
        
        if X_sparse is not None:
            total_energy = X_sparse.power(2).sum()
            metrics['SVD_Energy'] = {
                "captured_energy": float(cum_energy[-1]),
                "total_energy": float(total_energy),
                "ratio": float(cum_energy[-1] / total_energy)
            }
            cum_energy_ratio = cum_energy / total_energy
        else:
            cum_energy_ratio = cum_energy / cum_energy[-1]
            
        if save_dir:
            with open(os.path.join(save_dir, 'lightlira_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

        # Heatmaps (Skip if lightweight)
        if not lightweight:
            # Heatmap (Subset)
            sample_size = min(1000, M)
            plt.figure(figsize=(10, 8))
            S_sample = (self.V_k[:sample_size] * self.filter_diag) @ self.V_k.t()[:, :sample_size]
            if self.normalize:
                S_sample = self.S_diag_inv[:sample_size].unsqueeze(1) * S_sample * self.S_diag_inv[:sample_size].unsqueeze(0)
            vmax = np.percentile(np.abs(S_sample.cpu().numpy()), 99)
            sns.heatmap(S_sample.cpu().numpy(), cmap='viridis', vmax=vmax, vmin=-vmax/10)
            plt.title(f"LightLIRA Item-Item Matrix S (Subset {sample_size}x{sample_size})")
            if save_dir: plt.savefig(os.path.join(save_dir, 'lightlira_S_heatmap.png'))
            plt.close()
            
            # C Heatmap (Subset)
            sample_size_c = min(1000, N)
            plt.figure(figsize=(10, 8))
            inv_vals = 1.0 / (self.singular_values.pow(2) + self.reg_lambda)
            C_sample = (self.U_k[:sample_size_c] * inv_vals.unsqueeze(0)) @ self.U_k.t()[:, :sample_size_c]
            vmax_c = np.percentile(np.abs(C_sample.cpu().numpy()), 99)
            sns.heatmap(C_sample.cpu().numpy(), cmap='viridis', vmax=vmax_c, vmin=-vmax_c/10)
            plt.title(f"LightLIRA User-User Matrix C (Subset {sample_size_c}x{sample_size_c})")
            if save_dir: plt.savefig(os.path.join(save_dir, 'lightlira_C_heatmap.png'))
            plt.close()

            # Uk Heatmap
            plt.figure(figsize=(12, 6))
            Uk_viz = self.U_k[::max(1, N // 2000), :].detach().cpu().numpy()
            sns.heatmap(Uk_viz.T, cmap='viridis')
            plt.title("User Singular Vectors $U_k$")
            if save_dir: plt.savefig(os.path.join(save_dir, 'lightlira_Uk_heatmap.png'))
            plt.close()

        # Spectrum (Always generate)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(s_vals, label=r'$\sigma_k$')
        plt.yscale('log')
        plt.title("Spectrum")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(cum_energy_ratio, color='green')
        plt.title(f"Energy: {cum_energy_ratio[-1]*100:.1f}%")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(filter_w, color='orange')
        plt.title(fr"Filter ($\lambda={self.reg_lambda}$)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, 'lightlira_spectrum.png'))
        plt.close()


class AdaptiveLIRALayer(nn.Module):
    """
    Adaptive LIRA: Regularization depends on user activity.
    S = X.T @ (X @ X.T + lambda * diag(diag(X @ X.T)^alpha))^-1 @ X
    """
    def __init__(self, reg_lambda=500.0, alpha=0.5, normalize=True):
        super(AdaptiveLIRALayer, self).__init__()
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.normalize = normalize
        self.register_buffer('S', torch.empty(0))

    @torch.no_grad()
    def build(self, X_sparse):
        device = self.S.device
        
        print(f"[Ada-LIRA] Building on {device} (λ={self.reg_lambda}, α={self.alpha}) ..")
        X = torch.from_numpy(X_sparse.toarray()).float().to(device)
        n_users, n_items = X.shape
        
        # 1. Compute Kernel K = X @ X.T
        K = torch.mm(X, X.t())
        
        # 2. Extract diagonal D = diag(K)
        D = K.diag()
        
        # 3. Adaptive Regularization matrix
        # Handle alpha as list or scalar
        if isinstance(self.alpha, list):
            alpha_t = torch.tensor(self.alpha, device=device)
            reg_diag = self.reg_lambda * torch.pow(D, alpha_t)
        else:
            reg_diag = self.reg_lambda * torch.pow(D, self.alpha)
        
        # 4. Invert K_reg = K + Reg
        K_reg = K + torch.diag(reg_diag)
        C = torch.linalg.inv(K_reg)
        
        # 5. Compute S = X.T @ C @ X
        S = torch.mm(torch.mm(X.t(), C), X)
        
        if self.normalize:
            d = S.abs().sum(dim=1)
            d[d == 0] = 1.0
            d_inv_sqrt = d.pow(-0.5)
            S = d_inv_sqrt.unsqueeze(1) * S * d_inv_sqrt.unsqueeze(0)
            print("[Ada-LIRA] Applied Symmetric Normalization.")
            
        self.S = S
        print(f"[Ada-LIRA] S built. Norm: {S.norm():.2f}")

    def forward(self, X_batch, mask_observed=True):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        
        scores = torch.mm(X_batch, self.S)
        return scores


class PairLIRALayer(nn.Module):
    """
    Pair-wise LIRA (Sharper LIRA): Derived from pair-wise metric learning.
    S = (A^2 + lambda * I)^-1 @ A^2 where A = X.T @ X
    Filter: sigma^4 / (sigma^4 + lambda)
    """
    def __init__(self, reg_lambda=1e5, normalize=True):
        super(PairLIRALayer, self).__init__()
        self.reg_lambda = reg_lambda
        self.normalize = normalize
        self.register_buffer('S', torch.empty(0))

    @torch.no_grad()
    def build(self, X_sparse):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[Pair-LIRA] Building with λ={self.reg_lambda} ..")
        X = torch.from_numpy(X_sparse.toarray()).float().to(device)
        
        # 1. Compute A = X.T @ X (Item-Item co-occurrence)
        A = torch.mm(X.t(), X)
        
        # 2. Compute A^2
        A2 = torch.mm(A, A)
        
        # 3. Regularize and Invert: S = (A^2 + lambda * I)^-1 @ A^2
        n_items = A.size(0)
        A2_reg = A2 + self.reg_lambda * torch.eye(n_items, device=device)
        
        # Try higher precision on CPU/CUDA, stay float32 on MPS
        if device == 'mps':
            S = torch.mm(torch.inverse(A2_reg), A2)
        else:
            S = torch.mm(torch.inverse(A2_reg.double()), A2.double()).float()
        
        if self.normalize:
            d = S.abs().sum(dim=1)
            d[d == 0] = 1.0
            d_inv_sqrt = d.pow(-0.5)
            S = d_inv_sqrt.unsqueeze(1) * S * d_inv_sqrt.unsqueeze(0)
            print("[Pair-LIRA] Applied Symmetric Normalization.")
            
        self.S = S
        print(f"[Pair-LIRA] S built. Norm: {S.norm():.2f}")

    def forward(self, X_batch):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        return torch.mm(X_batch, self.S)
