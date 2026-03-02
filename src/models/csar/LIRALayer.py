import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.lira_visualizer import LIRAVisualizer

class _LIRAGraphCache:
    """
    Module-level memory cache for Normalized Graph Shift Operator (S_tilde).
    Keyed by (shape, nnz, data_checksum) to reuse across HPO trials in the same process.
    """
    _cache = {}
    
    @classmethod
    def _key(cls, X_sparse):
        if not sp.issparse(X_sparse):
            return None
        d = X_sparse.data
        checksum = hash((X_sparse.shape, X_sparse.nnz, 
                        d[:5].tobytes() if len(d) >= 5 else b'',
                        d[-5:].tobytes() if len(d) >= 5 else b''))
        return checksum
    
    @classmethod
    def get(cls, X_sparse, mode='normalized'):
        key = cls._key(X_sparse)
        if key is None: return None
        return cls._cache.get((mode, key))
    
    @classmethod
    def put(cls, X_sparse, S_sp, mode='normalized'):
        key = cls._key(X_sparse)
        if key is not None:
            cls._cache[(mode, key)] = S_sp
            print(f"[LIRA] Cached {mode} graph operator! Memory saved for future trials.")
    
    @classmethod
    def clear(cls):
        cls._cache.clear()

class LIRALayer(nn.Module):
    """
    LIRA - Linear Interest covariance Ridge Analysis (Dual Ridge Regression)
    """
    def __init__(self, reg_lambda=500.0, normalize=True):
        super(LIRALayer, self).__init__()
        # Handle list-wrapped hyperparams from search
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.normalize = normalize

        self.register_buffer('S', torch.empty(0))           # full-rank (Raw)

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

        # 1. Build Filter with RAW X
        X_raw = torch.from_numpy(X_sparse.toarray()).float().to(device)
        n_users, n_items = X_raw.shape

        # Gram Matrix K_raw = X_raw @ X_raw.T
        K_raw = torch.mm(X_raw, X_raw.t())
        self.gram_matrix = K_raw
        
        # Regularize: K_reg = K_raw + lambda * I
        K_reg = K_raw + self.reg_lambda * torch.eye(n_users, device=device)

        # Solve for C = K_reg^-1, then S = X.T @ C @ X
        # Instead of materializing C, solve K_reg @ (C @ X) = X for C @ X
        CX = torch.linalg.solve(K_reg, X_raw)  # (N, M)
        del K_reg
        
        # S = X.T @ CX
        S = torch.mm(X_raw.t(), CX)
        del CX
        
        self.S = S
        print(f"[LIRA] Raw S built (No Normalization). Norm: {S.norm():.2f}")


    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.S.numel() == 0:
            raise RuntimeError("build() must be called first")

        # Inference Item-only Prediction: X_batch @ S (where S already includes Di^-0.5)
        scores = torch.mm(X_batch, self.S)
            
        return scores

    def visualize_matrices(self, X_sparse=None, save_dir=None):
        S_raw_tensor = self.S_raw if hasattr(self, 'S_raw') else None
        gram_tensor = self.gram_matrix if hasattr(self, 'gram_matrix') else (self.C if hasattr(self, 'C') else None)
        LIRAVisualizer.visualize_dense_lira(S=self.S, K_Gram=gram_tensor, S_raw=S_raw_tensor, save_dir=save_dir)


class LightLIRALayer(nn.Module):
    """
    LightLIRA - Scalable Low-rank LIRA (SVD-based)
    Optimized for memory efficiency via batch-wise processing.
    """
    def __init__(self, k=200, reg_lambda=500.0, normalize=True):
        super(LightLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.normalize = normalize

        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))    # Raw right vectors
        
    @property
    def V_k(self):
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[LIRA] Building Raw Model (No Norm) with k={self.k}, λ={self.reg_lambda} ..")

        # 1. Perform SVD on Raw X
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        
        self.singular_values = s.to(device)
        self.V_raw = v.to(device)
        
        # Standard Filter: f = s^2 / (s^2 + lambda)
        s2 = self.singular_values.pow(2)
        self.filter_diag = s2 / (s2 + self.reg_lambda)
            
        print("[LIRA] Finished building Raw LowRank.")

        
        # 2. Apply spectral filter
        latent = latent * self.filter_diag
        
        # 3. Reconstruct back using raw item vectors
        scores = torch.mm(latent, self.V_raw.t())
        
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_svd_lira(
            self.singular_values, 
            self.filter_diag, 
            self.reg_lambda, 
            X_sparse=X_sparse, 
            save_dir=save_dir, 
            file_prefix='lightlira'
        )


class SpectralTikhonovLIRALayer(nn.Module):
    """
    Spectral Tikhonov LIRA
    Uses Tikhonov regularization with popularity decay: lambda_k = alpha * sigma_k^beta
    Resulting Filter:
    h(sigma_k) = sigma_k^(2-beta) / (sigma_k^(2-beta) + alpha)
    
    beta controls popularity penalty:
      - beta = 0: Standard Wiener filter (LightLIRA)
      - 0 < beta < 2: Penalizes high popularity (large sigma_k)
      - beta = 2: Completely flat filter 1/(1+alpha)
    """
    def __init__(self, k=200, alpha=500.0, beta=1.0, target_energy=0.99):
        super(SpectralTikhonovLIRALayer, self).__init__()
        self.k = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta = float(beta[0] if isinstance(beta, (list, np.ndarray)) else beta)
        self.target_energy = float(target_energy[0] if isinstance(target_energy, (list, np.ndarray)) else target_energy)

        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))
        
    @property
    def V_k(self):
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[LIRA] Building Spectral Model with {self.target_energy*100:.1f}% Energy SVD, α={self.alpha}, β={self.beta} ..")

        # 1. Perform SVD on Raw X
        from src.utils.gpu_accel import SVDCacheManager
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=None, target_energy=self.target_energy, dataset_name=dataset_name)
        
        self.k = len(s) # Update to auto-selected k
        
        self.singular_values = s.to(device)
        self.V_raw = v.to(device)
        
        # Spectral Tikhonov Filter: f = s^(2-beta) / (s^(2-beta) + alpha)
        s_pow = torch.pow(self.singular_values, 2.0 - self.beta)
        self.filter_diag = s_pow / (s_pow + self.alpha)
            
        print("[LIRA] Finished building Spectral Tikhonov Filter.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0:
            raise RuntimeError("build() must be called first")

        # (X @ V) * F @ V.T
        XV = torch.mm(X_batch, self.V_raw)           # (batch, k)
        XV_filtered = XV * self.filter_diag          # (batch, k) element-wise
        scores = torch.mm(XV_filtered, self.V_raw.t()) # (batch, items)
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, 
            self.filter_diag, 
            self.alpha,
            self.beta,
            X_sparse=X_sparse, 
            save_dir=save_dir, 
            file_prefix='spectral_tikhonov'
        )


class PowerLIRALayer(nn.Module):
    """
    LIRA with element-wise power sharpening on S.
    S = sgn(S) * |S|^p
    """
    def __init__(self, reg_lambda=500.0, power=2.0, threshold=1e-6):
        super(PowerLIRALayer, self).__init__()
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        val_power = power[0] if isinstance(power, (list, np.ndarray)) else power
        self.power = float(val_power)
        val_thresh = threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold
        self.threshold = float(val_thresh)
        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'

        print(f"[PowerLIRA] Building S using gpu_gram_solve (Item-Item space) ...")
        
        # 1. Compute P = (X^T X + lambda I)^-1 using the highly optimized EASE solver
        from src.utils.gpu_accel import gpu_gram_solve
        P_np = gpu_gram_solve(X_sparse, self.reg_lambda)
        
        # 2. Compute S = I - lambda * P (Woodbury identity for X^T (X X^T + \lambda I)^-1 X)
        n_items = X_sparse.shape[1]
        S_np = -self.reg_lambda * P_np
        np.fill_diagonal(S_np, S_np.diagonal() + 1.0)
        del P_np

        S = torch.from_numpy(S_np).float().to(device)
        del S_np
        
        # 3. Apply Power Sharpening
        S_sharpened = torch.sign(S) * torch.pow(torch.abs(S), self.power)
        
        # 4. Apply Epsilon Thresholding
        mask = torch.abs(S_sharpened) >= self.threshold
        S_final = S_sharpened * mask.float()
        
        # Calculate Sparsity
        n_elements = S_final.numel()
        n_zeros = (S_final == 0).sum().item()
        sparsity = n_zeros / n_elements
        
        # 5. Convert to Sparse Tensor
        target = S.device
        if 'mps' in str(target).lower(): self.S_sparse = S_final.cpu().to_sparse().coalesce()
        else: self.S_sparse = S_final.cpu().to_sparse().to(target).coalesce()
        
        del S, S_sharpened, S_final
        
        print(f"[PowerLIRA] Applied power={self.power}, threshold={self.threshold}. Sparsity: {sparsity:.4f}")
        print(f"[PowerLIRA] S stored as sparse tensor on {self.S_sparse.device}: {self.S_sparse._nnz()} non-zero elements.")

    def forward(self, X, user_ids=None):
        device = X.device
        
        # Optimization: Serve purely on Dense MPS to match EASE speed
        if not hasattr(self, 'W_dense') or self.W_dense is None:
            self.W_dense = self.S_sparse.to_dense().to(device)
            print(f"[PowerLIRA] Converted W to Dense ({self.W_dense.shape}) on {device} for fast serving.")
            
        scores = torch.mm(X, self.W_dense)
        return scores

    def visualize_matrices(self, X_sparse, save_dir):
        LIRAVisualizer.visualize_sparse_lira(
            self.S_sparse, 
            save_dir=save_dir, 
            title_suffix=f"(p={self.power}, threshold={self.threshold})"
        )


class LightPowerLIRALayer(nn.Module):
    """
    LightPowerLIRA: 
    1. SVD Low-rank approximation of S
    2. Power sharpening
    3. Thresholding & Sparse conversion
    """
    def __init__(self, k=200, reg_lambda=500.0, power=2.0, threshold=1e-6):
        super(LightPowerLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        val_power = power[0] if isinstance(power, (list, np.ndarray)) else power
        self.power = float(val_power)
        val_thresh = threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold
        self.threshold = float(val_thresh)
        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[LightPowerLIRA] Building with k={self.k}, λ={self.reg_lambda}, p={self.power}, threshold={self.threshold} ..")

        # 1. Low-rank Spectrum via SVD
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        
        s = s.to(device)
        v = v.to(device) # [M, K]
        
        # Spectral Filter: filter = s^2 / (s^2 + lambda)
        filter_diag = s.pow(2) / (s.pow(2) + self.reg_lambda)
        
        # 2. Materialize Kernel S_approx = V @ diag(filter) @ V.T
        # For stability and memory, we perform sharpening on CPU if using MPS
        if 'mps' in str(v.device).lower():
            v_cpu = v.cpu()
            f_cpu = filter_diag.cpu()
            S_approx = torch.mm(v_cpu * f_cpu, v_cpu.t())
            S_sharpened = torch.sign(S_approx) * torch.pow(torch.abs(S_approx), self.power)
            mask = torch.abs(S_sharpened) >= self.threshold
            S_final = S_sharpened * mask.float()
            del v_cpu, f_cpu, S_approx, S_sharpened, mask
        else:
            S_approx = torch.mm(v * filter_diag, v.t())
            S_sharpened = torch.sign(S_approx) * torch.pow(torch.abs(S_approx), self.power)
            mask = torch.abs(S_sharpened) >= self.threshold
            S_final = S_sharpened * mask.float()
        
        # Calculate Sparsity
        n_elements = S_final.numel()
        n_zeros = (S_final == 0).sum().item()
        sparsity = n_zeros / n_elements if n_elements > 0 else 1.0
        
        # 5. Convert to Sparse Tensor
        target = device
        if 'mps' in str(target).lower(): self.S_sparse = S_final.cpu().to_sparse()
        else: self.S_sparse = S_final.cpu().to_sparse().to(target)
        
        # Cleanup
        if 'S_approx' in locals(): del S_approx
        if 'S_sharpened' in locals(): del S_sharpened
        if 'S_final' in locals(): del S_final
        del v, s, filter_diag
        
        print(f"[LightPowerLIRA] Applied power={self.power}, threshold={self.threshold}. Sparsity: {sparsity:.4f}")
        print(f"[LightPowerLIRA] S stored as sparse tensor: {self.S_sparse._nnz()} non-zero elements.")

    def forward(self, X, user_ids=None):
        # X @ S = (S @ X.T).T
        if self.S_sparse.device != X.device:
            # Efficient cross-device sparse mm (especially for MPS stability)
            scores = torch.sparse.mm(self.S_sparse, X.t().to(self.S_sparse.device)).t().to(X.device)
        else:
            scores = torch.sparse.mm(self.S_sparse, X.t()).t()
            
        return scores

    def visualize_matrices(self, X_sparse, save_dir):
        LIRAVisualizer.visualize_sparse_lira(
            self.S_sparse, 
            save_dir=save_dir, 
            title_suffix=f"Light (k={self.k}, p={self.power}, threshold={self.threshold})"
        )

class SpectralPowerLIRALayer(nn.Module):
    """
    SpectralPowerLIRA: 
    Applies power transformation directly to the eigenvalues in the spectral domain.
    1. SVD Low-rank approximation
    2. Spectral Power on Singular Values: s_new = s^power
    3. Filter computation using s_new
    """
    def __init__(self, k=200, reg_lambda=500.0, power=1.0):
        super(SpectralPowerLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        val_power = power[0] if isinstance(power, (list, np.ndarray)) else power
        self.power = float(val_power)

        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[SpectralPowerLIRA] Building with k={self.k}, λ={self.reg_lambda}, spectral_power={self.power} ..")

        # 1. Low-rank Spectrum via SVD
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        
        s = s.to(device)
        self.V_raw = v.to(device) # [M, K]
        
        # 2. Spectral Power Transformation
        # Apply element-wise power to the singular values.
        # Original EASE Gram eigenvalues are lambda_k = s_k^2.
        # Raising the singular values to power p means scaling the spectrum.
        self.singular_values = s.pow(self.power)
        
        # 3. Spectral Filter: filter = s_new^2 / (s_new^2 + lambda)
        s2 = self.singular_values.pow(2)
        self.filter_diag = s2 / (s2 + self.reg_lambda)
        
        
        del v, s, u
        
        print(f"[SpectralPowerLIRA] Applied spectral power={self.power}. Filter max: {self.filter_diag.max().item():.4f}, min: {self.filter_diag.min().item():.4f}")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        latent = torch.mm(X_batch, self.V_raw)
        latent = latent * self.filter_diag
        scores = torch.mm(latent, self.V_raw.t())
        
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_svd_lira(
            self.singular_values, 
            self.filter_diag, 
            self.reg_lambda, 
            X_sparse=X_sparse, 
            save_dir=save_dir, 
            file_prefix='spectralpower_lightlira'
        )

class TaylorLIRALayer(nn.Module):
    """
    Pure Taylor-LIRA:
    W_K = sum_{k=1}^{K} (-1)^{k-1} S^k / lambda^k
    where S = X^T X (no normalization)

    - No degree normalization
    - Removes diagonal of S before expansion
    - Applies power nonlinearity (p=1.0 is no-op)
    - Applies final threshold
    """

    def __init__(self, reg_lambda=500.0, power=1.0, threshold=0.0, K=2):
        super().__init__()
        import numpy as np

        self.reg_lambda = float(reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda)
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.threshold = float(threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold)
        self.K = int(K[0] if isinstance(K, (list, np.ndarray)) else K)

        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        import scipy.sparse as sp
        import time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[TaylorLIRA] Building with λ={self.reg_lambda}, K={self.K}, p={self.power}, threshold={self.threshold} ..")

        start_time = time.time()
        
        calc_device = 'cpu' if 'mps' in str(device).lower() else device

        # Check in-memory cache first
        if not sp.issparse(X_sparse):
            X_sparse = sp.csr_matrix(X_sparse.cpu().numpy() if isinstance(X_sparse, torch.Tensor) else X_sparse)
            
        S_sp = _LIRAGraphCache.get(X_sparse, mode='normalized')
        
        if S_sp is None:
            print("[TaylorLIRA] Constructing S = X^T X using SciPy...")
            
            # Calculate item degrees D for normalization
            X_sp = X_sparse.tocsr()
            item_degrees = np.array(X_sp.sum(axis=0)).flatten()
            d_inv_sqrt = np.power(item_degrees, -0.5, where=item_degrees>0)
            d_inv_sqrt[item_degrees == 0] = 0.0
            
            # Create diagonal matrix D^{-1/2} and normalize X_tilde = X * D^{-1/2}
            D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            X_tilde_sp = X_sp.dot(D_inv_sqrt_mat)
            
            S_sp = X_tilde_sp.T.dot(X_tilde_sp)
            _LIRAGraphCache.put(X_sparse, S_sp, mode='normalized')
        else:
            print("[TaylorLIRA] Loaded S = X^T X from memory cache! Skipping construction.")
        
        # 3. Neumann expansion utilizing MPS / GPU dense multiplications when available
        print(f"[TaylorLIRA] Applying Neumann expansion (K={self.K}) on {calc_device} ...")
        
        # Helper function for per-hop sparse thresholding
        def threshold_sp(matrix, eps):
            if eps <= 0: return matrix
            matrix.data[np.abs(matrix.data) < eps] = 0
            matrix.eliminate_zeros()
            return matrix

        # Base Matrix S^1
        S_power_sp = S_sp.copy()
        S_power_sp = threshold_sp(S_power_sp, self.threshold)
        W_sp = None
        
        # If K > 1, convert base S to dense on device for fast MatMul
        S_dense_device = None
        if self.K > 1:
            S_coo = S_sp.tocoo()
            indices = torch.from_numpy(np.vstack((S_coo.row, S_coo.col))).long()
            values = torch.from_numpy(S_coo.data).float()
            S_dense_device = torch.sparse_coo_tensor(indices, values, S_coo.shape).to_dense().to(calc_device)

        for k in range(1, self.K + 1):
            coef = ((-1) ** (k - 1)) / (self.reg_lambda ** k)
            term_sp = S_power_sp.copy()
            term_sp.data *= coef

            if W_sp is None:
                W_sp = term_sp
            else:
                W_sp = W_sp + term_sp

            if k < self.K:
                print(f"[TaylorLIRA] Constructing {k+1}-Hop Term on {calc_device} ...")
                
                # Perform fast Dense matrix multiplication natively on GPU/MPS
                S_power_coo = S_power_sp.tocoo()
                indices = torch.from_numpy(np.vstack((S_power_coo.row, S_power_coo.col))).long()
                values = torch.from_numpy(S_power_coo.data).float()
                
                S_power_sparse_device = torch.sparse_coo_tensor(indices, values, S_power_coo.shape).to(calc_device)
                S_power_dense_device = S_power_sparse_device.to_dense()
                
                # fast Native GPU matmul
                next_S_dense = torch.mm(S_power_dense_device, S_dense_device)
                
                # Truncate and back to sparse
                mask = torch.abs(next_S_dense) >= self.threshold
                next_S_dense = next_S_dense * mask
                
                next_S_sparse = next_S_dense.to_sparse().cpu()
                
                # Convert back to SciPy for W_sp accumulation natively in CPU RAM
                indices = next_S_sparse.indices().numpy()
                values = next_S_sparse.values().numpy()
                S_power_sp = sp.csr_matrix((values, (indices[0], indices[1])), shape=next_S_sparse.shape)
                
                del S_power_sparse_device, S_power_dense_device, next_S_dense, next_S_sparse
                if 'cuda' in str(calc_device): torch.cuda.empty_cache()

        # Convert back to PyTorch
        W_sp = W_sp.tocoo()
        idx_W = torch.from_numpy(np.vstack((W_sp.row, W_sp.col))).long()
        val_W = torch.from_numpy(W_sp.data).float()
        
        W = torch.sparse_coo_tensor(idx_W, val_W, W_sp.shape)
        self.S_sparse = W.to('cpu' if 'mps' in str(device).lower() else device).coalesce()
        
        elapsed = time.time() - start_time
        print(f"[TaylorLIRA] Done in {elapsed:.2f}s")
        print(f"[TaylorLIRA] nnz = {self.S_sparse._nnz()}")

    def forward(self, X_batch, user_ids=None):
        device = X_batch.device
        
        # Optimization: We can perform the matrix multiplication with scipy if both are cpu
        # But for pytorch, since EASE is dense and FAST on MPS/GPU, we can try to do the same here
        # Convert S_sparse to Dense and perform Dense x Dense multiplication natively on MPS/GPU
        # We will cache the dense weight matrix so it doesn't happen per batch
        
        if not hasattr(self, 'W_dense') or self.W_dense is None:
            # We delay dense conversion until inference to save memory during build
            self.W_dense = self.S_sparse.to_dense().to(device)
            print(f"[TaylorLIRA] Converted W to Dense ({self.W_dense.shape}) on {device} for fast serving.")
            
        scores = torch.mm(X_batch, self.W_dense)
        return scores


def cg_solve_batch(matvec_func, B, max_iter=50, tol=1e-6):
    """
    Solves A X = B implicitly using the matvec_func.
    B: (N, batch_size) shape
    """
    X = torch.zeros_like(B)
    R = B - matvec_func(X)
    P = R.clone()

    Rs_old = torch.sum(R * R, dim=0)

    for i in range(max_iter):
        AP = matvec_func(P)
        # Add eps to avoid division by zero
        alpha = Rs_old / (torch.sum(P * AP, dim=0) + 1e-12)

        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)

        Rs_new = torch.sum(R * R, dim=0)

        if torch.max(torch.sqrt(Rs_new)) < tol:
            break

        beta = Rs_new / (Rs_old + 1e-12)
        P = R + P * beta.unsqueeze(0)
        Rs_old = Rs_new

    return X


class CGLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, max_iter=30, tol=1e-6):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.X_sparse = None
        self.X_sparse_t = None

    def build(self, X_sparse, dataset_name=None):
        import scipy.sparse as sp
        import time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'

        print(f"[CGLIRA] Building with λ={self.reg_lambda}, max_iter={self.max_iter}, tol={self.tol} ..")
        start_time = time.time()

        # If MPS is used, keep sparse tensors on CPU to avoid NotImplementedError for addmm
        target_device = 'cpu' if 'mps' in str(device).lower() else device

        if sp.issparse(X_sparse):
            X_coo = X_sparse.tocoo()
            indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
            values = torch.from_numpy(X_coo.data).float()
            shape = X_coo.shape
            self.X_sparse = torch.sparse_coo_tensor(indices, values, shape).to(target_device)
        else:
            self.X_sparse = X_sparse.to(target_device)

        self.X_sparse = self.X_sparse.coalesce()
        self.X_sparse_t = self.X_sparse.t().coalesce()

        elapsed = time.time() - start_time
        print(f"[CGLIRA] Built in {elapsed:.2f}s!")

    def matvec(self, V):
        """
        Compute (S + lambda I) V
        where S = X^T X
        V shape: (num_items, batch_size)
        """
        # Ensure V is on same device as X_sparse
        V = V.to(self.X_sparse.device)
        # X V
        tmp = torch.sparse.mm(self.X_sparse, V)
        # X^T (X V)
        SV = torch.sparse.mm(self.X_sparse_t, tmp)
        return SV + self.reg_lambda * V

    def forward(self, X_batch, user_ids=None):
        """
        Solves (S + λI) Z = S X_batch^T
        """
        target_device = X_batch.device
        S_device = self.X_sparse.device

        # Transpose X_batch for column-wise operations
        x_t = X_batch.t().to(S_device)

        # y = S x_u = X^T (X x_t)
        tmp = torch.sparse.mm(self.X_sparse, x_t)
        y = torch.sparse.mm(self.X_sparse_t, tmp)

        # solve (S + lambda I) Z = y
        Z = cg_solve_batch(self.matvec, y, max_iter=self.max_iter, tol=self.tol)

        # Return (batch_size, num_items)
        return Z.t().to(target_device)


from numpy.polynomial.chebyshev import Chebyshev

class ChebyshevLIRALayer(nn.Module):
    """
    Filter-Perspective Chebyshev-LIRA.
    1. Defines the Graph Shift Operator (Normalized S_tilde).
    2. Approximates the purely inverse Covariance Filter (C) using Chebyshev (FIR Filter).
    3. Applies Non-linear Power Sharpening to the Filter C.
    4. Convolves the Filter with the Shift Operator (W = S_tilde * C_powered) for O(1) serving.
    """
    def __init__(self, reg_lambda=500.0, power=2.0, threshold=1e-6, K=3):
        super(ChebyshevLIRALayer, self).__init__()
        self.reg_lambda = float(reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda)
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.threshold = float(threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold)
        self.K = int(K[0] if isinstance(K, (list, np.ndarray)) else K)
        
        # 최종 서빙을 위한 가중치 필터 (W_final = S_tilde * C_powered)
        self.W_sparse = None 

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        print(f"[FilterLIRA] Building λ={self.reg_lambda}, K={self.K}, p={self.power} ..")
        start_time = time.time()
        
        calc_device = 'cpu' if 'mps' in str(device).lower() else device

        # ---------------------------------------------------------
        # Phase 1: Graph Shift Operator 정의 (S_tilde) using SciPy
        # ---------------------------------------------------------
        if not sp.issparse(X_sparse):
            X_sparse = sp.csr_matrix(X_sparse.cpu().numpy() if isinstance(X_sparse, torch.Tensor) else X_sparse)
            
        S_tilde_sp = _LIRAGraphCache.get(X_sparse, mode='normalized')
        
        if S_tilde_sp is None:
            print("[FilterLIRA] Phase 1: Constructing Graph Shift Operator (S_tilde) with SciPy...")
            
            X_sp = X_sparse.tocsr()
            n_items = X_sp.shape[1]
            
            # Calculate item degrees D for normalization
            item_degrees = np.array(X_sp.sum(axis=0)).flatten()
            d_inv_sqrt = np.power(item_degrees, -0.5, where=item_degrees>0)
            d_inv_sqrt[item_degrees == 0] = 0.0
            
            # Create diagonal matrix D^{-1/2} and normalize X_tilde = X * D^{-1/2}
            D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            X_tilde_sp = X_sp.dot(D_inv_sqrt_mat)
            
            # S_tilde = X_tilde^T * X_tilde
            S_tilde_sp = X_tilde_sp.T.dot(X_tilde_sp)
            _LIRAGraphCache.put(X_sparse, S_tilde_sp, mode='normalized')
        else:
            print("[FilterLIRA] Phase 1: Loaded Graph Shift Operator (S_tilde) from memory cache!")
            n_items = S_tilde_sp.shape[1]
            
        # ---------------------------------------------------------
        # Phase 2: FIR Filter 설계 (Chebyshev Approximation of C) using SciPy
        # ---------------------------------------------------------
        print("[FilterLIRA] Phase 2: Designing FIR Wiener Filter (C_approx) with SciPy...")
        def target_filter_func(x_mapped):
            eig = (x_mapped + 1.0) / 2.0
            return 1.0 / (eig + self.reg_lambda) # 순정 위너 필터 C
        
        cheb_approx = Chebyshev.interpolate(target_filter_func, deg=self.K)
        c = cheb_approx.coef  

        I_sp = sp.eye(n_items, format='csr')
        
        S_mapped_sp = (S_tilde_sp * 2.0) - I_sp
        T_k_minus_2_sp = I_sp
        C_approx_sp = T_k_minus_2_sp * c[0]
        T_k_minus_1_sp = S_mapped_sp
        C_approx_sp = C_approx_sp + T_k_minus_1_sp * c[1]
        
        for k in range(2, self.K + 1):
            term_sp = S_mapped_sp.dot(T_k_minus_1_sp) * 2.0
            T_k_sp = term_sp - T_k_minus_2_sp
            C_approx_sp = C_approx_sp + T_k_sp * c[k]
            T_k_minus_2_sp = T_k_minus_1_sp
            T_k_minus_1_sp = T_k_sp

        # ---------------------------------------------------------
        # Phase 3: 비선형 필터 진화 및 합성 (W = S_tilde * C_powered)
        # ---------------------------------------------------------
        print("[FilterLIRA] Phase 3: Non-linear Activation & Final Convolution (SciPy)...")
        
        # 필터(C)에 파워 샤프닝을 먹여 증폭기(+)와 억제기(-)의 텐션을 극대화!
        C_approx_sp.data = np.sign(C_approx_sp.data) * np.power(np.abs(C_approx_sp.data), self.power)

        # 합성 필터 연산: W_final = S_tilde * C_powered (SciPy)
        W_final_sp = S_tilde_sp.dot(C_approx_sp)
        
        # Helper function for sparse thresholding
        def threshold_sp(matrix, eps):
            if eps <= 0: return matrix
            matrix.data[np.abs(matrix.data) < eps] = 0
            matrix.eliminate_zeros()
            return matrix

        # 최종 노이즈 가지치기 (Thresholding)
        W_final_sp = threshold_sp(W_final_sp, self.threshold)
        
        sparsity = 1.0 - (W_final_sp.nnz / (n_items * n_items))
        
        # Convert W_final back to PyTorch Sparse Tensor
        W_final_sp = W_final_sp.tocoo()
        idx_W = torch.from_numpy(np.vstack((W_final_sp.row, W_final_sp.col))).long()
        val_W = torch.from_numpy(W_final_sp.data).float()
        
        target_device = device
        if 'mps' in str(target_device).lower():
            self.W_sparse = torch.sparse_coo_tensor(idx_W, val_W, (n_items, n_items)).to('cpu').coalesce()
        else:
            self.W_sparse = torch.sparse_coo_tensor(idx_W, val_W, (n_items, n_items)).to(target_device).coalesce()
        
        elapsed = time.time() - start_time
        print(f"[FilterLIRA] Built in {elapsed:.2f}s! Final Sparsity: {sparsity:.6f}")

    def forward(self, X_batch, user_ids=None):
        # Serving은 O(1) Matrix Multiplication으로 빛처럼 빠르게!
        target_device = X_batch.device
        
        # MPS sparse mm workaround
        calc_device = 'cpu' if 'mps' in str(target_device).lower() else target_device

        if self.W_sparse.device != calc_device:
            scores = torch.sparse.mm(
                self.W_sparse.to(calc_device), 
                X_batch.t().to(calc_device)
            ).t().to(target_device)
        else:
            scores = torch.sparse.mm(
                self.W_sparse, 
                X_batch.t().to(calc_device)
            ).t().to(target_device)
            
        return scores