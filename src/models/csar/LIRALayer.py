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
    
    @classmethod
    def clear(cls):
        cls._cache.clear()

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
        X_raw = torch.from_numpy(X_sparse.toarray()).float().to(device)
        n_users, n_items = X_raw.shape
        K_raw = torch.mm(X_raw, X_raw.t())
        self.gram_matrix = K_raw
        K_reg = K_raw + self.reg_lambda * torch.eye(n_users, device=device)
        CX = torch.linalg.solve(K_reg, X_raw)  
        del K_reg
        S = torch.mm(X_raw.t(), CX)
        del CX
        self.S = S
        print(f"[{self.__class__.__name__}] Raw S built. Norm: {S.norm():.2f}")

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
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        self.singular_values = s.to(device)
        self.V_raw = v.to(device)
        s2 = self.singular_values.pow(2)
        self.filter_diag = s2 / (s2 + self.reg_lambda)
        print(f"[{self.__class__.__name__}] Finished building Raw LowRank.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0: raise RuntimeError("build() first")
        latent = torch.mm(X_batch, self.V_raw)           
        latent = latent * self.filter_diag               
        return torch.mm(latent, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_svd_lira(self.singular_values, self.filter_diag, self.reg_lambda, X_sparse=X_sparse, save_dir=save_dir, file_prefix='lightlira')


class ASPIRELayer(nn.Module):
    def __init__(self, k=200, alpha=500.0, beta=1.0, target_energy=0.99):
        super(ASPIRELayer, self).__init__()
        self.k = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta = float(beta[0] if isinstance(beta, (list, np.ndarray)) else beta)
        self.target_energy = float(target_energy[0] if isinstance(target_energy, (list, np.ndarray)) else target_energy)
        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))
        
    @property
    def V_k(self): return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=None, target_energy=self.target_energy, dataset_name=dataset_name)
        self.k = len(s)
        self.singular_values = s.to(device)
        self.V_raw = v.to(device)
        s_pow = torch.pow(self.singular_values, 2.0 - self.beta)
        self.filter_diag = s_pow / (s_pow + self.alpha)
        print(f"[{self.__class__.__name__}] Finished building Spectral Tikhonov Filter.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0: raise RuntimeError("build() first")
        XV = torch.mm(X_batch, self.V_raw)
        XV_filtered = XV * self.filter_diag
        return torch.mm(XV_filtered, self.V_raw.t())


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
        if 'mps' in str(S.device).lower(): self.S_sparse = S_final.cpu().to_sparse().coalesce()
        else: self.S_sparse = S_final.cpu().to_sparse().to(S.device).coalesce()
        print(f"[{self.__class__.__name__}] Power={self.power} build complete.")

    def forward(self, X, user_ids=None):
        if not hasattr(self, 'W_dense') or self.W_dense is None:
            self.W_dense = self.S_sparse.to_dense().to(X.device)
        return torch.mm(X, self.W_dense)


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
        self.S_sparse = S_final.to_sparse().coalesce()
        if 'cuda' in str(device): self.S_sparse = self.S_sparse.to(device)

    def forward(self, X, user_ids=None):
        if self.S_sparse.device != X.device:
            return torch.sparse.mm(self.S_sparse, X.t().to(self.S_sparse.device)).t().to(X.device)
        return torch.sparse.mm(self.S_sparse, X.t()).t()


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
        self.singular_values = s.pow(self.power).to(device)
        self.V_raw = v.to(device)
        s2 = self.singular_values.pow(2)
        self.filter_diag = s2 / (s2 + self.reg_lambda)

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
        idx_W = torch.from_numpy(np.vstack((W_sp.row, W_sp.col))).long()
        val_W = torch.from_numpy(W_sp.data).float()
        self.S_sparse = torch.sparse_coo_tensor(idx_W, val_W, W_sp.shape).to('cpu' if 'mps' in str(device).lower() else device).coalesce()

    def forward(self, X_batch, user_ids=None):
        if not hasattr(self, 'W_dense') or self.W_dense is None:
            self.W_dense = self.S_sparse.to_dense().to(X_batch.device)
        return torch.mm(X_batch, self.W_dense)


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
        self.W_sparse = None

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
        idx = torch.from_numpy(np.vstack((W_coo.row, W_coo.col))).long()
        val = torch.from_numpy(W_coo.data).float()
        self.W_sparse = torch.sparse_coo_tensor(idx, val, W_final.shape).to('cpu' if 'mps' in str(device).lower() else device).coalesce()

    def forward(self, X_batch, user_ids=None):
        # [IMPROVEMENT 2] OOM 폭탄 제거! to_dense() 없이 Sparse 행렬곱으로 처리
        # W_sparse는 대칭행렬이므로 X @ W = (W @ X.T).T 와 동치입니다.
        device = X_batch.device
        sparse_dev = self.W_sparse.device
        
        X_batch_t = X_batch.t().to(sparse_dev) # (Items, Batch)
        out_t = torch.sparse.mm(self.W_sparse, X_batch_t) # Sparse(M,M) @ Dense(M,B) -> Dense(M,B)
        
        return out_t.t().to(device) # 다시 (Batch, Items)로 복구


class ChebyASPIRELayer(nn.Module):
    def __init__(self, alpha=500.0, degree=20, beta=0.5, lambda_max_estimate='auto', threshold=1e-4):
        super().__init__()
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.beta = float(beta)
        self.lambda_max_estimate = lambda_max_estimate
        self.threshold = float(threshold)
        
        self.register_buffer('cheby_coeffs', torch.empty(0))
        self.register_buffer('t_mid', torch.tensor(0.0))
        self.register_buffer('t_half', torch.tensor(0.0))
        self.X_sparse_torch = None
        self.X_sparse_torch_t = None

    def _aspire_filter(self, lam):
        # 수학적 증명 확인 완료: lambda는 X^TX의 고유값(σ^2). 
        # 따라서 lam^(0.75)는 σ^(1.5)와 정확히 일치합니다.
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
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        
        self.X_sparse_torch = torch.sparse_coo_tensor(indices, values, X_coo.shape).to(self.sparse_device).coalesce()
        self.X_sparse_torch_t = self.X_sparse_torch.t().coalesce()

        if self.lambda_max_estimate == 'auto':
            lambda_max = self._estimate_lambda_max(self.X_sparse_torch, self.X_sparse_torch_t)
        else:
            lambda_max = float(self.lambda_max_estimate)
            
        lambda_min = 0.0
        
        coeffs = self._compute_chebyshev_coeffs(lambda_min, lambda_max, self.degree)
        self.cheby_coeffs = torch.from_numpy(coeffs).float().to(device)
        self.t_mid = torch.tensor((lambda_max + lambda_min) / 2.0, device=device)
        self.t_half = torch.tensor((lambda_max - lambda_min) / 2.0, device=device)
        
        print(f"[{self.__class__.__name__}] Build complete. coefficients computed.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.X_sparse_torch is None:
            raise RuntimeError("build() first")

        device = X_batch.device
        coeffs = self.cheby_coeffs.to(device)
        t_mid = self.t_mid.to(device)
        t_half = self.t_half.to(device)
        sparse_dev = self.X_sparse_torch.device
        
        def S_mapped(v):
            v_sp = v.to(sparse_dev)
            res = torch.sparse.mm(self.X_sparse_torch_t, torch.sparse.mm(self.X_sparse_torch, v_sp))
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