import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager
import time


class SVD_AE(BaseModel):
    """
    SVD-AE (Simple Autoencoders for Collaborative Filtering) - IJCAI 2024
    Reference: Hong et al., https://github.com/seoyoungh/svd-ae

    논문 수식:
        objective:  min_B  ||R - R_tilde @ B||_F^2
        closed-form:  B* = R_tilde^+ @ R = V_k @ Σ_k^{-1} @ Q_k^T @ R
        inference:  r_hat_u = r_u @ B* = r_u @ V_k @ P_k

        where:
            R_tilde = D_U^{-1/2} R D_I^{-1/2}  (symmetric normalization)
            R_tilde ≈ Q_k Σ_k V_k^T             (truncated SVD)
            P_k = Σ_k^{-1} @ Q_k^T @ R          (k x n_items, precomputed)

    Fix: Q_k^T @ R 를 dense @ sparse 가 아니라 sparse @ dense 방향으로 계산
         (dense @ sparse 는 R 을 암묵적으로 dense 변환 → OOM)
    """

    def __init__(self, config, data_loader):
        super(SVD_AE, self).__init__(config, data_loader)

        self.gamma = config['model'].get('gamma', 0.04)
        if isinstance(self.gamma, list):
            self.gamma = self.gamma[0]

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        raw_m = int(round(self.gamma * min(self.n_users, self.n_items)))
        max_k = min(self.n_users, self.n_items) - 1
        self.k = max(1, min(raw_m, max_k))

        self.train_matrix_csr = None

        self.svd_manager = SVDCacheManager(device=self.device.type)
        self.register_cache_manager('svd', self.svd_manager)

        self.register_buffer('V_k', torch.empty(0))  # (n_items, k)
        self.register_buffer('P_k', torch.empty(0))  # (k, n_items)

        self._log(f"Initialized (gamma={self.gamma} -> k={self.k})")

    def fit(self, data_loader):
        self._log(f"Fitting (k={self.k})...")
        start = time.time()

        # 1. Sparse interaction matrix R
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        R = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        self.train_matrix_csr = R

        # 2. Symmetric normalization → R_tilde
        self._log("Normalizing matrix...")
        d_U = np.array(R.sum(axis=1)).flatten()
        d_I = np.array(R.sum(axis=0)).flatten()
        d_U[d_U == 0] = 1.0
        d_I[d_I == 0] = 1.0
        d_U_inv_sqrt = np.power(d_U, -0.5)
        d_I_inv_sqrt = np.power(d_I, -0.5)

        R_tilde = sp.diags(d_U_inv_sqrt) @ R @ sp.diags(d_I_inv_sqrt)

        # 3. Truncated SVD on R_tilde
        dataset_name = self.config.get('dataset_name', 'unknown')
        self._log(f"SVD (k={self.k})...")
        Q_k_t, Sigma_k_t, V_k_t, _ = self.svd_manager.get_svd(
            R_tilde, k=self.k, dataset_name=dataset_name
        )
        Q_k    = Q_k_t.to(self.device).float()      # (n_users, k)
        Sigma_k = Sigma_k_t.to(self.device).float()  # (k,)
        self.V_k = V_k_t.to(self.device).float()     # (n_items, k)

        del R_tilde

        # 4. P_k = Σ_k^{-1} @ Q_k^T @ R   shape: (k, n_items)
        #
        # [Fix] 반드시 sparse @ dense 방향으로 계산해야 OOM 없음
        #   잘못된 방향: Q_k_np.T @ R  → numpy가 R을 dense 변환 → OOM
        #   올바른 방향: R.T @ Q_k_np  → scipy sparse @ dense → (n_items, k)
        #               → .T           → (k, n_items)
        self._log("Computing P_k = Σ_k^{-1} @ Q_k^T @ R ...")
        Q_k_np = Q_k.cpu().numpy()                           # (n_users, k)
        Q_k_proj_np = np.array(R.T @ Q_k_np, dtype=np.float32).T  # (k, n_items)
        # R.T: (n_items, n_users) sparse  @  Q_k_np: (n_users, k) dense
        # → (n_items, k) dense  →  .T  →  (k, n_items)

        Q_k_proj = torch.from_numpy(Q_k_proj_np).to(self.device)
        Sigma_k_inv = 1.0 / (Sigma_k + 1e-10)
        self.P_k = Sigma_k_inv.unsqueeze(1) * Q_k_proj  # (k, n_items)

        del Q_k_np, Q_k_proj_np, Q_k_proj

        self._log(f"Fitted in {time.time()-start:.1f}s  "
                  f"| V_k: {self.V_k.shape}, P_k: {self.P_k.shape}")

    def forward(self, user_ids, item_ids=None):
        """
        r_hat_u = r_u @ V_k @ P_k
                = r_u @ V_k @ (Σ_k^{-1} Q_k^T R)
        """
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.asarray(user_ids)

        X_batch = torch.from_numpy(
            self.train_matrix_csr[u_ids_np].toarray()
        ).float().to(self.device)                      # (B, n_items)

        latent = X_batch @ self.V_k                    # (B, k)
        scores = latent @ self.P_k                     # (B, n_items)

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)

        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def get_final_item_embeddings(self):
        return self.V_k

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None
