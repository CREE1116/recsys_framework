import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager
import time


class SVD_AE(BaseModel):
    """
    SVD-AE (Simple Autoencoders for Collaborative Filtering) - IJCAI 2024
    Reference: Hong et al., "SVD-AE: Simple Autoencoders for Collaborative Filtering"

    Theoretical Logic:
    1. Problem: min_B || R - R_tilde @ B ||_2^2
       where R_tilde = D_U^{-1/2} R D_I^{-1/2} is the Normalized Adjacency Matrix.
    2. Closed-form Solution via Pseudo-inverse: B = R_tilde^+ @ R
    3. Low-rank Inductive Bias (Truncated SVD):
       R_tilde ~ Q_m @ Sigma_m @ V_m^T
       Thus, R_tilde^+ ~ V_m @ Sigma_m^{-1} @ Q_m^T
    4. Inference: R_hat = R @ B = (R @ V_m) @ (Sigma_m^{-1} Q_m^T R)

    [Optimization] Uses SVDCacheManager for caching SVD results across HPO trials.
    The cache key includes gamma (rank ratio) and norm_weight to avoid collision.
    """
    def __init__(self, config, data_loader):
        super(SVD_AE, self).__init__(config, data_loader)

        # User defined hyperparameter: ratio of min(|U|, |I|) to use for SVD rank
        self.gamma = config['model'].get('gamma', 0.04)

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        # Calculate dynamic rank m = floor(gamma * min(|U|, |I|))
        raw_m = int(round(self.gamma * min(self.n_users, self.n_items)))
        max_k = min(self.n_users, self.n_items) - 1
        self.k = max(1, min(raw_m, max_k))

        self.train_matrix_csr = None

        # SVDCacheManager for caching R_tilde SVD
        self.svd_manager = SVDCacheManager(device=self.device.type)

        # We store the decoupled components to avoid materializing the dense M x M matrix B
        self.register_buffer('V_k', torch.empty(0))  # M x k
        self.register_buffer('P_k', torch.empty(0))  # k x M

        self._log(f"Initialized (gamma={self.gamma} -> k={self.k})")

    def fit(self, data_loader):
        self._log(f"Fitting (k={self.k})...")
        start_time = time.time()

        # 1. Build Sparse Interaction Matrix R (X)
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)

        R = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        self.train_matrix_csr = R

        # 2. Compute Degree Matrices D_U^{-1/2} and D_I^{-1/2}
        self._log("Computing Normalized Adjacency Matrix R_tilde...")
        d_U = np.array(R.sum(axis=1)).flatten()
        d_I = np.array(R.sum(axis=0)).flatten()

        d_U[d_U == 0] = 1.0  # Prevent div by zero
        d_I[d_I == 0] = 1.0

        d_U_inv_sqrt = np.power(d_U, -0.5)
        d_I_inv_sqrt = np.power(d_I, -0.5)

        D_U_inv_sqrt = sp.diags(d_U_inv_sqrt)
        D_I_inv_sqrt = sp.diags(d_I_inv_sqrt)
        R_tilde = D_U_inv_sqrt.dot(R).dot(D_I_inv_sqrt)

        # 3. Truncated SVD via SVDCacheManager (caching + MPS acceleration)
        # Note: cache key = dataset_name + "_svdae_gamma{gamma}"
        # - rank k is derived from gamma, so same gamma = same SVD for same dataset
        dataset_name = self.config.get('dataset_name', 'unknown_svdae')
        cache_key = f"{dataset_name}_svdae_g{self.gamma:.4f}"

        self._log(f"Performing Truncated SVD via SVDCacheManager (k={self.k})...")
        # SVDCacheManager.get_svd(matrix, k, dataset_name) returns (U, S, V, energy)
        # U: (n_users, k), S: (k,), V: (n_items, k)
        Q_k_t, Sigma_k_t, V_k_t, energy = self.svd_manager.get_svd(
            R_tilde, k=self.k, dataset_name=cache_key
        )
        # SVDCacheManager returns: u:(N,k), s:(k,), v:(M,k)
        Q_k = Q_k_t.to(self.device).float()     # N x k
        Sigma_k = Sigma_k_t.to(self.device).float()  # k
        V_k = V_k_t.to(self.device).float()     # M x k

        self.V_k = V_k
        del R_tilde, D_U_inv_sqrt, D_I_inv_sqrt

        # 4. Precompute P_k = Sigma_k^{-1} @ Q_k^T @ R  =>  (k x M)
        self._log("Constructing decoupled inference matrix P_k...")

        Q_k_np = Q_k.cpu().numpy()           # N x k
        # Q_k^T @ R  -> (k x N) @ (N x M) -> (k x M)
        Q_k_proj_np = Q_k_np.T @ R           # scipy sparse handles this

        Q_k_proj = torch.from_numpy(Q_k_proj_np.astype(np.float32)).to(self.device)
        Sigma_k_inv = 1.0 / (Sigma_k + 1e-10)

        # P_k = diag(Sigma_k_inv) @ Q_k_proj
        self.P_k = Sigma_k_inv.unsqueeze(1) * Q_k_proj  # k x M

        del Q_k_np, Q_k_proj_np, Q_k_proj, Sigma_k_inv

        elapsed = time.time() - start_time
        self._log(f"SVD-AE fitted. Energy captured (σ² sum): {energy_sq:.4f}, Energy captured (σ sum): {energy_linear:.4f}")
        print(f"         V_k: {self.V_k.shape}, P_k: {self.P_k.shape}")

    def forward(self, user_ids, item_ids=None):
        if self.V_k is None or self.P_k is None:
            raise RuntimeError("[SVD-AE] Model not fitted.")

        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids

        user_input_sparse = self.train_matrix_csr[u_ids_np]
        X_batch = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)

        # Two-Step Projection: R_hat = (R @ V_k) @ P_k
        latent_batch = torch.mm(X_batch, self.V_k)    # Batch x k
        scores = torch.mm(latent_batch, self.P_k)      # Batch x M

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            batch_indices = torch.arange(len(user_ids), device=self.device)
            return scores[batch_indices, item_ids]

        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        return self.V_k

    def get_embeddings(self):
        return None, self.V_k
