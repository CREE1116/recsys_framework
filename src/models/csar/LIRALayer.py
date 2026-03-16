import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.lira_visualizer import LIRAVisualizer




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
        n_users, n_items = X_sparse.shape
        if torch.cuda.is_available(): dev = 'cuda'
        elif torch.backends.mps.is_available(): dev = 'mps'
        else: dev = 'cpu'
        
        calc_dev = dev
        
        # [OPTIMIZATION] Switch between Primal (Item-Item) and Dual (User-User) form
        # chooses the smaller dimension for inversion to save O(N^3) time and O(N^2) memory.
        if n_items <= n_users:
            print(f"[{self.__class__.__name__}] Using Primal Form (Item-Item: {n_items}x{n_items}) since I <= U")
            # Convert to dense (only items)
            X_dense = torch.from_numpy(X_sparse.toarray()).float().to(calc_dev)
            # G = X^T X
            G = torch.mm(X_dense.t(), X_dense)
            # G_reg = G + λI
            G_target = G.clone() # Keep original G for the rhs
            G.diagonal().add_(self.reg_lambda)
            
            from src.utils.gpu_accel import gpu_cholesky_solve
            # S = (X^T X + λI)^-1 (X^T X)
            S = gpu_cholesky_solve(G.cpu().numpy(), G_target.cpu().numpy(), device=calc_dev, return_tensor=True)
            del X_dense, G, G_target
        else:
            print(f"[{self.__class__.__name__}] Using Dual Form (User-User: {n_users}x{n_users}) since U < I")
            X_dense = torch.from_numpy(X_sparse.toarray()).float().to(calc_dev)
            # K = X X^T
            K = torch.mm(X_dense, X_dense.t())
            # K_reg = K + λI
            K.diagonal().add_(self.reg_lambda)
            
            from src.utils.gpu_accel import gpu_cholesky_solve
            # CX = (X X^T + λI)^-1 X
            CX = gpu_cholesky_solve(K.cpu().numpy(), X_dense.cpu().numpy(), device=calc_dev, return_tensor=True)
            # S = X^T CX
            S = torch.mm(X_dense.t(), CX)
            del X_dense, K, CX

        self.register_buffer('S', S.to(dev))
        print(f"[{self.__class__.__name__}] Build complete. Calculation Device: {calc_dev}, Model Device: {dev}")

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
        dev = self.singular_values.device
        manager = SVDCacheManager(device=dev)
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
        # Upgrade to rich spectral visualization (shared with ASPIRE)
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, self.filter_diag, 
            alpha=self.reg_lambda, beta=0.0, 
            X_sparse=X_sparse, save_dir=save_dir, file_prefix='lightlira'
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
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
        from src.utils.gpu_accel import gpu_gram_solve
        P_np = gpu_gram_solve(X_sparse, self.reg_lambda)
        S_np = -self.reg_lambda * P_np
        np.fill_diagonal(S_np, S_np.diagonal() + 1.0)
        del P_np
        S = torch.from_numpy(S_np).float().to(device)
        S_sharpened = torch.sign(S) * torch.pow(torch.abs(S), self.power)
        if self.threshold > 0:
            mask = torch.abs(S_sharpened) >= self.threshold
            S_sharpened = S_sharpened * mask.float()
        
        self.register_buffer('S', S_sharpened)
        print(f"[{self.__class__.__name__}] Power={self.power} build complete (DENSE) on {device}.")

    def forward(self, X, user_ids=None):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        return torch.mm(X, self.S)


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
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        s, v = s.to(device), v.to(device)
        filter_diag = s.pow(2) / (s.pow(2) + self.reg_lambda)
        if device == 'mps':
            v_cpu, f_cpu = v.cpu(), filter_diag.cpu()
            S_approx = torch.mm(v_cpu * f_cpu, v_cpu.t())
        else:
            S_approx = torch.mm(v * filter_diag, v.t())
        S_sharpened = torch.sign(S_approx) * torch.pow(torch.abs(S_approx), self.power)
        if self.threshold > 0:
            mask = torch.abs(S_sharpened) >= self.threshold
            S_sharpened = S_sharpened * mask.float()
        
        self.register_buffer('S', S_sharpened)
        print(f"[{self.__class__.__name__}] LightPowerLIRA build complete (DENSE).")

    def forward(self, X, user_ids=None):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        return torch.mm(X, self.S)


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
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
        manager = SVDCacheManager(device=device)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        self.register_buffer('singular_values', s.pow(self.power).to(device))
        self.register_buffer('V_raw', v.to(device))
        s2 = self.singular_values.pow(2)
        self.register_buffer('filter_diag', s2 / (s2 + self.reg_lambda))
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        # Power-aware spectral visualization
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, self.filter_diag, 
            alpha=self.reg_lambda, beta=0.0, a=self.power, 
            X_sparse=X_sparse, save_dir=save_dir, file_prefix='specpower'
        )

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
        self.register_buffer('S', torch.empty(0))

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
        calc_device = 'cpu' if device == 'mps' else device
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
        
        W_final = W_sp.toarray()
        self.register_buffer('S', torch.from_numpy(W_final).float())
        print(f"[{self.__class__.__name__}] TaylorLIRA build complete (DENSE).")

    def forward(self, X_batch, user_ids=None):
        if self.S.numel() == 0: raise RuntimeError("build() first")
        return torch.mm(X_batch, self.S)


class CGLIRALayer(nn.Module):
    def __init__(self, reg_lambda=500.0, max_iter=30, tol=1e-6):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol

    def build(self, X_sparse, dataset_name=None):
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
        target_device = 'cpu' if device == 'mps' else device
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

