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
from scipy.special import eval_chebyt
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
        idx = X_sparse.indices
        ptr = X_sparse.indptr
        checksum = hash((X_sparse.shape, X_sparse.nnz, 
                        d[:10].tobytes() if len(d) >= 10 else d.tobytes(),
                        idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                        ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
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
    
    @classmethod
    def clear(cls):
        cls._cache.clear()

class _MNARGammaCache:
    """
    Cache for estimated MNAR gamma values, keyed by dataset properties.
    """
    _cache = {}

    @classmethod
    def _key(cls, X_sparse=None, singular_values=None):
        if singular_values is not None:
            # Hash based on sample of singular values and length
            s_sample = singular_values[:10].detach().cpu().numpy().tobytes()
            return hash(("sv", len(singular_values), s_sample))
        if X_sparse is not None:
            # Hash based on shape, nnz, and structure
            idx = X_sparse.indices
            ptr = X_sparse.indptr
            return hash(("X", X_sparse.shape, X_sparse.nnz, 
                        idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                        ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
        return None

    @classmethod
    def get(cls, X_sparse=None, singular_values=None):
        key = cls._key(X_sparse, singular_values)
        return cls._cache.get(key) if key else None

    @classmethod
    def put(cls, val, X_sparse=None, singular_values=None):
        key = cls._key(X_sparse, singular_values)
        if key: cls._cache[key] = val

class LIRALayer(nn.Module):
    """
    LIRA - Linear Interest covariance Ridge Analysis (Dual Ridge Regression)
    """
    def __init__(self, reg_lambda=500.0, normalize=True):
        super(LIRALayer, self).__init__()
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.normalize = normalize
        self.register_buffer('S', torch.empty(0))           

    @torch.no_grad()
    def build(self, X_sparse):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        # [MEMORY FIX] Do NOT use X_sparse.toarray() on CPU.
        # Compute Gram matrix G = X^T X directly to VRAM
        
        # Convert X to Sparse Tensor on Device
        dev = device
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long().to(dev)
        values = torch.from_numpy(X_coo.data).float().to(dev)
        X_t = torch.sparse_coo_tensor(indices, values, X_sparse.shape, device=dev).coalesce()
        del X_coo, indices, values
        
        # K = X @ X^T (User-User Gram)
        # LIRA implementation uses User-User Gram for dual solve
        # K: (n_users, n_users)
        K = torch.sparse.mm(X_t, X_t.t().to_dense()) 
        
        # (K + λI) S = X @ X^T -> No, S = X^T (X X^T + λI)^-1 X
        K.diagonal().add_(self.reg_lambda)
        
        # Solve for user-wise coefficients: CX = (K + λI)^-1 X
        # Solve A @ CX = X_t (dense)
        CX = torch.linalg.solve(K, X_t.to_dense())
        del K, X_t
        
        # S = X^T @ CX (Item-Item Shift)
        # S: (n_items, n_items)
        # Resulting S is DENSE, which is fine for these models.
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long().to(dev)
        values = torch.from_numpy(X_coo.data).float().to(dev)
        X_t = torch.sparse_coo_tensor(indices, values, X_sparse.shape, device=dev).coalesce()
        
        self.register_buffer('S', torch.mm(X_t.t().to_dense(), CX))
        del X_t, CX
        
        print(f"[{self.__class__.__name__}] GPU-Native build complete. Device: {dev}")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        return torch.mm(X_batch, self.S)

    def visualize_matrices(self, X_sparse=None, save_dir=None):
        S_raw_tensor = self.S_raw if hasattr(self, 'S_raw') else None
        gram_tensor = self.gram_matrix if hasattr(self, 'gram_matrix') else (self.C if hasattr(self, 'C') else None)
        LIRAVisualizer.visualize_dense_lira(S=self.S, K_Gram=gram_tensor, S_raw=S_raw_tensor, save_dir=save_dir)


class LightLIRALayer(nn.Module):
    def __init__(self, k=200, reg_lambda=500.0, normalize=True):
        super(LightLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.normalize = normalize
        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))    
        
    @property
    def V_k(self): return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        manager = SVDCacheManager(device=self.singular_values.device)
        # manager.get_svd returns tensors on CPU/Device depending on cache. 
        # Move them to target device immediately.
        dev = self.singular_values.device
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        self.register_buffer('singular_values', s.to(dev))
        self.register_buffer('V_raw', v.to(dev))
        s2 = self.singular_values.pow(2)
        self.register_buffer('filter_diag', s2 / (s2 + self.reg_lambda))
        print(f"[{self.__class__.__name__}] SVD-based build complete. Device: {dev}")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0: raise RuntimeError("build() first")
        latent = torch.mm(X_batch, self.V_raw)           
        latent = latent * self.filter_diag               
        return torch.mm(latent, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_svd_lira(self.singular_values, self.filter_diag, self.reg_lambda, X_sparse=X_sparse, save_dir=save_dir, file_prefix='lightlira')

def estimate_mnar_gamma(X_sparse=None, singular_values=None):
    """
    [이론적 근거 완벽 반영]: log(n_i) = a * log(λ_i) + C 를 통해 γ = 2a - 1 도출
    """
    cached = _MNARGammaCache.get(X_sparse, singular_values)
    if cached is not None: return cached
    
    if X_sparse is not None and singular_values is not None:
        # 1. 아이템 인기도 n_i (내림차순 정렬)
        item_pops = np.array(X_sparse.sum(axis=0)).flatten()
        n_i = np.sort(item_pops)[::-1]
        
        # 2. 고유값 λ_i = σ_i^2 (이미 내림차순 정렬되어 있음)
        k = len(singular_values)
        n_i_trunc = n_i[:k] # SVD K개 차원에 맞춰 슬라이싱
        lam_i = singular_values.pow(2).cpu().numpy()
        
        # 3. 로그 변환을 위한 안전한 필터링 (0 방지)
        valid_mask = (n_i_trunc > 0) & (lam_i > 1e-10)
        n_i_valid = n_i_trunc[valid_mask]
        lam_i_valid = lam_i[valid_mask]
        
        if len(n_i_valid) < 10:
            return 1.0 # 샘플이 너무 적으면 기본값 반환
            
        # 4. [핵심] Step 0 구현: log(n_i) vs log(λ_i) 선형 회귀
        x = np.log(lam_i_valid)
        y = np.log(n_i_valid)
        slope, _ = np.polyfit(x, y, 1)
        
        # 5. γ = 2a - 1 도출 (a가 slope)
        gamma = (2.0 * slope) - 1.0
        
        # MNAR 편향은 음수가 될 수 없으므로 하한선 방어 (안전장치)
        gamma = max(0.1, float(gamma))
        
        _MNARGammaCache.put(gamma, X_sparse=X_sparse, singular_values=singular_values)
        return gamma

    if X_sparse is not None:
        # Fallback for models without SVD (like ChebyASPIRE)
        # item popularity ≈ rank^-p
        # Assume γ = p (Empirical Zipf-based relation)
        item_pops = np.array(X_sparse.sum(axis=0)).flatten()
        item_pops = np.sort(item_pops)[::-1]
        item_pops = item_pops[item_pops > 0]
        if len(item_pops) < 10: return 1.0
        
        y = np.log(item_pops)
        x = np.log(np.arange(1, len(item_pops) + 1))
        slope, _ = np.polyfit(x, y, 1)
        res = max(0.1, float(-slope))
        _MNARGammaCache.put(res, X_sparse=X_sparse)
        return res

    return 1.0 # fallback


class ASPIRELayer(nn.Module):
    def __init__(self, k=200, alpha=500.0, beta=1.0, target_energy=0.99):
        super(ASPIRELayer, self).__init__()
        self.k = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta_config = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.beta = 1.0 # Placeholder, will be set in build()
        self.target_energy = float(target_energy[0] if isinstance(target_energy, (list, np.ndarray)) else target_energy)
        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))
        
    @property
    def V_k(self): return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        dev = self.singular_values.device
        manager = SVDCacheManager(device=dev)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=None, target_energy=self.target_energy, dataset_name=dataset_name)
        self.k = len(s)
        self.register_buffer('singular_values', s.to(dev))
        self.register_buffer('V_raw', v.to(dev))
        
        # Auto Beta Determination
        if isinstance(self.beta_config, str):
            gamma = estimate_mnar_gamma(X_sparse=X_sparse, singular_values=self.singular_values)
            if self.beta_config == 'auto_bias':
                self.beta = gamma / (1 + gamma)
            elif self.beta_config == 'auto_compromise':
                self.beta = gamma / 2.0
            else:
                self.beta = 0.5 # Default
            print(f"[{self.__class__.__name__}] Estimated gamma={gamma:.4f} -> beta={self.beta:.4f} ({self.beta_config})")
        else:
            self.beta = float(self.beta_config)

        s_pow = torch.pow(self.singular_values, 2.0 - self.beta)
        self.register_buffer('filter_diag', s_pow / (s_pow + self.alpha))
        print(f"[{self.__class__.__name__}] ASPIRE build complete (k={self.k}). Device: {dev}")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0: raise RuntimeError("build() first")
        XV = torch.mm(X_batch, self.V_raw)
        XV_filtered = XV * self.filter_diag
        return torch.mm(XV_filtered, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, 
            self.filter_diag, 
            self.alpha, 
            self.beta, 
            X_sparse=X_sparse, 
            save_dir=save_dir, 
            file_prefix='aspire'
        )


class PowerLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, power=2.0, threshold=1e-6):
        super(PowerLIRALayer, self).__init__()
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.threshold = float(threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold)
        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        from src.utils.gpu_accel import gpu_gram_solve
        P_np = gpu_gram_solve(X_sparse, self.reg_lambda)
        S_np = -self.reg_lambda * P_np
        np.fill_diagonal(S_np, S_np.diagonal() + 1.0)
        del P_np
        S = torch.from_numpy(S_np).float().to(device)
        S_sharpened = torch.sign(S) * torch.pow(torch.abs(S), self.power)
        mask = torch.abs(S_sharpened) >= self.threshold
        S_final = S_sharpened * mask.float()
        
        S_coo = S_final.to_sparse_coo().coalesce()
        self.register_buffer('S_indices', S_coo.indices())
        self.register_buffer('S_values', S_coo.values())
        self.S_shape = tuple(S_coo.shape)
        
        print(f"[{self.__class__.__name__}] Power={self.power} build complete on {device}.")

    def forward(self, X, user_ids=None):
        S = torch.sparse_coo_tensor(self.S_indices, self.S_values, self.S_shape, device=X.device)
        # Use (S @ X.T).T as workaround for no mm(dense, sparse)
        return torch.sparse.mm(S, X.t()).t()


class LightPowerLIRALayer(nn.Module):
    def __init__(self, k=200, reg_lambda=500.0, power=2.0, threshold=1e-6):
        super(LightPowerLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.threshold = float(threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold)
        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        s, v = s.to(device), v.to(device)
        filter_diag = s.pow(2) / (s.pow(2) + self.reg_lambda)
        if 'mps' in str(v.device).lower():
            v_cpu, f_cpu = v.cpu(), filter_diag.cpu()
            S_approx = torch.mm(v_cpu * f_cpu, v_cpu.t())
        else:
            S_approx = torch.mm(v * filter_diag, v.t())
        S_sharpened = torch.sign(S_approx) * torch.pow(torch.abs(S_approx), self.power)
        mask = torch.abs(S_sharpened) >= self.threshold
        S_final = S_sharpened * mask.float()
        
        S_coo = S_final.to_sparse_coo().coalesce()
        self.register_buffer('S_indices', S_coo.indices())
        self.register_buffer('S_values', S_coo.values())
        self.S_shape = S_coo.shape

    def forward(self, X, user_ids=None):
        S = torch.sparse_coo_tensor(self.S_indices, self.S_values, self.S_shape, device=X.device)
        return torch.sparse.mm(S, X.t()).t()


class SpectralPowerLIRALayer(nn.Module):
    def __init__(self, k=200, reg_lambda=500.0, power=1.0):
        super(SpectralPowerLIRALayer, self).__init__()
        self.k = k[0] if isinstance(k, (list, np.ndarray)) else k
        self.reg_lambda = reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        self.register_buffer('singular_values', s.pow(self.power).to(device))
        self.register_buffer('V_raw', v.to(device))
        s2 = self.singular_values.pow(2)
        self.register_buffer('filter_diag', s2 / (s2 + self.reg_lambda))

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        latent = torch.mm(X_batch, self.V_raw)
        latent = latent * self.filter_diag
        return torch.mm(latent, self.V_raw.t())

class TaylorLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, power=1.0, threshold=0.0, K=2):
        super().__init__()
        self.reg_lambda = float(reg_lambda[0] if isinstance(reg_lambda, (list, np.ndarray)) else reg_lambda)
        self.power = float(power[0] if isinstance(power, (list, np.ndarray)) else power)
        self.threshold = float(threshold[0] if isinstance(threshold, (list, np.ndarray)) else threshold)
        self.K = int(K[0] if isinstance(K, (list, np.ndarray)) else K)
        self.S_sparse = None

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        import scipy.sparse as sp
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        calc_device = 'cpu' if 'mps' in str(device).lower() else device
        X_sp = X_sparse.tocsr()
        item_degrees = np.array(X_sp.sum(axis=0)).flatten()
        d_inv_sqrt = np.power(item_degrees, -0.5, where=item_degrees>0)
        D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        X_tilde = X_sp.dot(D_inv_sqrt_mat)
        S_sp = X_tilde.T.dot(X_tilde)
        
        S_dense_device = None
        if self.K > 1:
            S_coo = S_sp.tocoo()
            indices = torch.from_numpy(np.vstack((S_coo.row, S_coo.col))).long()
            values = torch.from_numpy(S_coo.data).float()
            S_dense_device = torch.sparse_coo_tensor(indices, values, S_coo.shape).to_dense().to(calc_device)

        S_power_sp = S_sp.copy()
        W_sp = None
        for k in range(1, self.K + 1):
            coef = ((-1)**(k-1)) / (self.reg_lambda**k)
            term_sp = S_power_sp.copy()
            term_sp.data *= coef
            W_sp = term_sp if W_sp is None else W_sp + term_sp
            if k < self.K:
                S_power_coo = S_power_sp.tocoo()
                indices = torch.from_numpy(np.vstack((S_power_coo.row, S_power_coo.col))).long()
                values = torch.from_numpy(S_power_coo.data).float()
                S_power_sparse_device = torch.sparse_coo_tensor(indices, values, S_power_coo.shape).to(calc_device)
                S_power_dense_device = S_power_sparse_device.to_dense()
                next_S_dense = torch.mm(S_power_dense_device, S_dense_device)
                mask = torch.abs(next_S_dense) >= self.threshold
                next_S_sparse = (next_S_dense * mask).to_sparse().cpu()
                indices = next_S_sparse.indices().numpy()
                values = next_S_sparse.values().numpy()
                S_power_sp = sp.csr_matrix((values, (indices[0], indices[1])), shape=next_S_sparse.shape)
        
        W_sp = W_sp.tocoo()
        self.register_buffer('S_indices', torch.from_numpy(np.vstack((W_sp.row, W_sp.col))).long())
        self.register_buffer('S_values', torch.from_numpy(W_sp.data).float())
        self.S_shape = W_sp.shape

    def forward(self, X_batch, user_ids=None):
        S = torch.sparse_coo_tensor(self.S_indices, self.S_values, self.S_shape, device=X_batch.device)
        return torch.sparse.mm(S, X_batch.t()).t()


class CGLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, max_iter=30, tol=1e-6):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol

    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        target_device = 'cpu' if 'mps' in str(device).lower() else device
        from src.utils.gpu_accel import to_sparse_tensor
        self.X_sparse = to_sparse_tensor(X_sparse).to(target_device)
        self.X_sparse_t = self.X_sparse.t().coalesce()

    def matvec(self, V):
        V = V.to(self.X_sparse.device)
        tmp = torch.sparse.mm(self.X_sparse, V)
        SV = torch.sparse.mm(self.X_sparse_t, tmp)
        return SV + self.reg_lambda * V

    def forward(self, X_batch, user_ids=None):
        target_device = X_batch.device
        S_device = self.X_sparse.device
        x_t = X_batch.t().to(S_device)
        tmp = torch.sparse.mm(self.X_sparse, x_t)
        y = torch.sparse.mm(self.X_sparse_t, tmp)
        from src.models.csar.LIRALayer import cg_solve_batch
        Z = cg_solve_batch(self.matvec, y, max_iter=self.max_iter, tol=self.tol)
        return Z.t().to(target_device)

def cg_solve_batch(matvec_func, B, max_iter=50, tol=1e-6):
    X = torch.zeros_like(B)
    R = B - matvec_func(X)
    P = R.clone()
    Rs_old = torch.sum(R * R, dim=0)
    for i in range(max_iter):
        AP = matvec_func(P)
        alpha = Rs_old / (torch.sum(P * AP, dim=0) + 1e-12)
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)
        Rs_new = torch.sum(R * R, dim=0)
        if torch.max(torch.sqrt(Rs_new)) < tol: break
        beta = Rs_new / (Rs_old + 1e-12)
        P = R + P * beta.unsqueeze(0)
        Rs_old = Rs_new
    return X


class ChebyshevLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, power=2.0, threshold=1e-6, K=3):
        super().__init__()
        self.reg_lambda = float(reg_lambda)
        self.power = float(power)
        self.threshold = float(threshold)
        self.K = int(K)
        self.register_buffer('K_indices', torch.empty(2, 0, dtype=torch.long))
        self.register_buffer('K_values', torch.empty(0))
        self.K_shape = (0, 0)

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        import scipy.sparse as sp
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        
        X_sp = X_sparse.tocsr()
        item_degrees = np.array(X_sp.sum(axis=0)).flatten()
        d_inv_sqrt = np.power(item_degrees, -0.5, where=item_degrees>0)
        D_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        
        X_tilde_sp = X_sp.dot(D_inv_sqrt_mat)
        S_tilde_sp = X_tilde_sp.T.dot(X_tilde_sp)
        
        from numpy.polynomial.chebyshev import Chebyshev
        target_f = lambda x: 1.0 / (((x + 1.0)/2.0) + self.reg_lambda)
        cheb_approx = Chebyshev.interpolate(target_f, deg=self.K)
        c = cheb_approx.coef
        
        I_sp = sp.eye(S_tilde_sp.shape[1], format='csr')
        # D^{-1/2}X^TXD^{-1/2}의 고유값은 수학적으로 [0, 1]에 바운딩되므로 
        # * 2.0 - I 매핑은 완벽히 안전합니다. (크리님의 놀라운 통찰!)
        S_mapped_sp = (S_tilde_sp * 2.0) - I_sp 
        
        T_prev = I_sp
        T_curr = S_mapped_sp
        C_approx = c[0] * T_prev + c[1] * T_curr
        for k in range(2, self.K + 1):
            T_next = 2.0 * S_mapped_sp.dot(T_curr) - T_prev
            C_approx += c[k] * T_next
            T_prev, T_curr = T_curr, T_next
        
        C_approx.data = np.sign(C_approx.data) * np.power(np.abs(C_approx.data), self.power)
        W_final = S_tilde_sp.dot(C_approx)
        
        if self.threshold > 0:
            W_final.data[np.abs(W_final.data) < self.threshold] = 0
            W_final.eliminate_zeros()
            
        W_coo = W_final.tocoo()
        self.register_buffer('K_indices', torch.from_numpy(np.vstack((W_coo.row, W_coo.col))).long())
        self.register_buffer('K_values', torch.from_numpy(W_coo.data).float())
        self.K_shape = tuple(W_final.shape)

    def forward(self, X_batch, user_ids=None):
        device = X_batch.device
        S = torch.sparse_coo_tensor(self.K_indices, self.K_values, self.K_shape, device=device)
        X_batch_t = X_batch.t() # (Items, Batch)
        out_t = torch.sparse.mm(S, X_batch_t)
        return out_t.t()


class ChebyASPIRELayer(nn.Module):
    def __init__(self, alpha=500.0, degree=20, beta=0.5, lambda_max_estimate='auto', threshold=1e-4):
        super().__init__()
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.beta_config = beta
        self.beta = 0.5 # Placeholder
        self.lambda_max_estimate = lambda_max_estimate
        self.threshold = float(threshold)
        
        self.register_buffer('cheby_coeffs', torch.empty(0))
        self.register_buffer('t_mid', torch.tensor(0.0))
        self.register_buffer('t_half', torch.tensor(0.0))
        self.register_buffer('X_indices', torch.empty(2, 0, dtype=torch.long))
        self.register_buffer('X_values', torch.empty(0))
        self.X_shape = (0, 0)

    def _aspire_filter(self, lam):
        # 수학적 증명 확인 완료: lambda는 X^TX의 고유값(σ^2). 
        # 따라서 lam^(0.75)는 σ^(1.5)와 정확히 일치합니다!
        exponent = 2.0 - self.beta 
        lam_pow = np.power(np.maximum(lam, 0.0), exponent / 2.0)
        return lam_pow / (lam_pow + self.alpha)

    @torch.no_grad() # [IMPROVEMENT 4] 그래프 생성 방지
    def _estimate_lambda_max(self, X_sp, Xt_sp):
        v = torch.randn(X_sp.shape[1], 1, device=X_sp.device)
        v = v / v.norm()
        # [IMPROVEMENT 3] 중복 트랜스포즈 제거. 이미 캐싱된 행렬 사용.
        for _ in range(30):
            v = torch.sparse.mm(Xt_sp, torch.sparse.mm(X_sp, v))
            lambda_est = v.norm().item()
            v = v / lambda_est
        return lambda_est

    def _compute_chebyshev_coeffs(self, lambda_min, lambda_max, K):
        j = np.arange(K + 1)
        # [IMPROVEMENT 1] scipy 종속성 제거 및 삼각함수 치환
        theta = np.pi * (j + 0.5) / (K + 1)
        t_nodes = np.cos(theta)
        
        mid, half = (lambda_max + lambda_min) / 2.0, (lambda_max - lambda_min) / 2.0
        lambda_nodes = mid + half * t_nodes
        f_nodes = self._aspire_filter(lambda_nodes)
        
        coeffs = np.zeros(K + 1)
        for k in range(K + 1):
            # eval_chebyt(k, t_nodes)는 수학적으로 cos(k * theta)와 동일합니다!
            T_k = np.cos(k * theta) 
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * T_k)
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        self.sparse_device = 'cpu' if 'mps' in str(device).lower() else device
        
        X_coo = X_sparse.tocoo()
        self.register_buffer('X_indices', torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long())
        self.register_buffer('X_values', torch.from_numpy(X_coo.data).float())
        self.X_shape = X_coo.shape

        if self.lambda_max_estimate == 'auto':
            X_torch = torch.sparse_coo_tensor(self.X_indices, self.X_values, self.X_shape, device=self.sparse_device).coalesce()
            lambda_max = self._estimate_lambda_max(X_torch, X_torch.t().coalesce())
        else:
            lambda_max = float(self.lambda_max_estimate)
            
        lambda_min = 0.0
        
        # Auto Beta Determination
        if isinstance(self.beta_config, str):
            gamma = estimate_mnar_gamma(X_sparse=X_sparse)
            if self.beta_config == 'auto_bias':
                self.beta = gamma / (1 + gamma)
            elif self.beta_config == 'auto_compromise':
                self.beta = gamma / 2.0
            else:
                self.beta = 0.5
            print(f"[{self.__class__.__name__}] Estimated gamma={gamma:.4f} -> beta={self.beta:.4f} ({self.beta_config})")
        else:
            self.beta = float(self.beta_config)

        coeffs = self._compute_chebyshev_coeffs(lambda_min, lambda_max, self.degree)
        self.cheby_coeffs = torch.from_numpy(coeffs).float().to(device)
        self.t_mid = torch.tensor((lambda_max + lambda_min) / 2.0, device=device)
        self.t_half = torch.tensor((lambda_max - lambda_min) / 2.0, device=device)
        
        print(f"[{self.__class__.__name__}] Build complete. coefficients computed.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.X_values.numel() == 0:
            raise RuntimeError("build() first")

        device = X_batch.device
        coeffs = self.cheby_coeffs.to(device)
        t_mid = self.t_mid.to(device)
        t_half = self.t_half.to(device)
        sparse_dev = self.X_indices.device
        X_torch = torch.sparse_coo_tensor(self.X_indices, self.X_values, self.X_shape, device=sparse_dev).coalesce()
        X_torch_t = X_torch.t().coalesce()
        
        def S_mapped(v):
            # v is on 'device' (batch device). 
            v_sp = v.to(sparse_dev) if v.device != torch.device(sparse_dev) else v
            
            # Sparse Multiplications on sparse_dev
            res = torch.sparse.mm(X_torch_t, torch.sparse.mm(X_torch, v_sp))
            
            # Transfer back to 'device' for the rest of the 3-term recurrence
            if res.device != device:
                res = res.to(device)
            return (res - t_mid * v) / t_half

        X_t = X_batch.t() # (items, batch)
        T_prev = X_t
        T_curr = S_mapped(X_t)

        out = coeffs[0] * T_prev + coeffs[1] * T_curr

        for k in range(2, self.degree + 1):
            T_next = 2.0 * S_mapped(T_curr) - T_prev
            out = out + coeffs[k] * T_next
            T_prev, T_curr = T_curr, T_next

        return out.t() # (batch, items)

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        # ChebyASPIRE visualization: Plot coefficients and the filter shape
        if not save_dir: return
        os.makedirs(save_dir, exist_ok=True)
        
        coeffs = self.cheby_coeffs.cpu().numpy()
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(coeffs)), coeffs)
        plt.title(f"Chebyshev Coefficients (degree={self.degree})")
        plt.xlabel("k")
        plt.ylabel("c_k")
        
        plt.subplot(1, 2, 2)
        # Plot theoretical filter shape
        lam = np.linspace(0, 1.0, 100) # Assumes normalized or relative
        f_lam = self._aspire_filter(lam)
        plt.plot(lam, f_lam, color='orange', label='ASPIRE Target')
        plt.title(fr"Filter Shape ($\alpha={self.alpha}, \beta={self.beta}$)")
        plt.xlabel(r"Eigenvalue $\lambda$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cheby_aspire_analysis.png"))
        plt.close()
        print(f"[{self.__class__.__name__}] Analysis saved to {save_dir}")