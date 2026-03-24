import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager


class GF_CF(BaseModel):
    """
    GF-CF: Graph Filter based Collaborative Filtering (CIKM 2021)
    Reference: https://github.com/yshenaw/GF_CF

    논문 eq.(22):
        s_u = r_u @ (R_tilde^T @ R_tilde  +  alpha * D_I^{-1/2} V V^T D_I^{1/2})

    where:
        R_tilde = D_U^{-1/2} R D_I^{-1/2}   (symmetric normalization)
        V       = top-K right singular vectors of R_tilde  (shape: n_items x K)

    공식 구현(getUsersRating) 대응:
        U_2 = r_u @ norm_adj.T @ norm_adj              <- linear term
        U_1 = r_u @ d_mat_i @ V @ V^T @ d_mat_i_inv   <- ideal low-pass term
        ret = U_2 + alpha * U_1

    메모리 최적화:
        - N x N 크기의 dense 가중치 행렬 W를 직접 생성하지 않음.
        - Low-rank property (X @ V) @ V.T 를 활용하여 연산 복잡도 최적화.
    """

    def __init__(self, config, data_loader):
        super(GF_CF, self).__init__(config, data_loader)

        k = self.config['model'].get('k', 256)
        self.k = k[0] if isinstance(k, list) else k
        self.alpha = self.config['model'].get('alpha', 0.3)

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        self.svd_manager = SVDCacheManager(device=self.device.type)
        self.register_cache_manager('svd', self.svd_manager)

        # fit() 이후 채워지는 변수들
        self.train_matrix_csr = None  # raw R           (sparse, n_users x n_items)
        self.norm_adj          = None  # R_tilde          (sparse, n_users x n_items)
        self.V                 = None  # (n_items, k) right singular vectors
        self.d_mat_i           = None  # D_I^{-1/2}      (1-D tensor, n_items)
        self.d_mat_i_inv       = None  # D_I^{+1/2}      (1-D tensor, n_items)

        self._log(f"Initialized (k={self.k}, alpha={self.alpha})")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, data_loader):
        self._log("Building interaction matrix...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        vals = np.ones(len(rows), dtype=np.float32)

        R = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        self.train_matrix_csr = R

        # 1. Symmetric normalization -> R_tilde
        self._log("Normalizing matrix (Symmetric)...")

        rowsum = np.array(R.sum(axis=1)).flatten()
        d_inv_row = np.power(rowsum, -0.5)
        d_inv_row[np.isinf(d_inv_row)] = 0.
        D_U_inv_sqrt = sp.diags(d_inv_row)

        colsum = np.array(R.sum(axis=0)).flatten()
        d_inv_col = np.power(colsum, -0.5)
        d_inv_col[np.isinf(d_inv_col)] = 0.

        norm_adj = D_U_inv_sqrt.dot(R).dot(sp.diags(d_inv_col))
        self.norm_adj = norm_adj.tocsc()

        # D_I scaling vectors: 1-D, for element-wise multiply in forward()
        d_mat_i_inv_vals = np.where(d_inv_col > 0, 1.0 / d_inv_col, 0.0).astype(np.float32)
        self.d_mat_i     = torch.from_numpy(d_inv_col.astype(np.float32)).to(self.device)
        self.d_mat_i_inv = torch.from_numpy(d_mat_i_inv_vals).to(self.device)

        # 2. SVD on R_tilde -> top-K right singular vectors V (n_items, k)
        dataset_name = self.config.get('dataset_name', 'unknown')
        self._log(f"Performing SVD (k={self.k})...")
        u, s, v, _ = self.svd_manager.get_svd(self.norm_adj, k=self.k, dataset_name=dataset_name)

        # get_svd returns v of shape (n_items, k)
        self.V = v[:, :self.k].to(self.device).float()  # (n_items, k)

        # Sanity checks
        self._log(f"V shape: {self.V.shape}")
        self._log(f"Top-5 singular values: {s[:5].tolist()}")
        if self.k > 1:
            ortho_err = torch.norm(self.V.t() @ self.V - torch.eye(self.k, device=self.device))
            self._log(f"V orthogonality error ||V^T V - I||: {ortho_err:.6f}")

        self._log("Fit complete.")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, user_ids, item_ids=None):
        """
        공식 구현 getUsersRating()과 동일한 로직:
            U_2 = r_u @ R_tilde^T @ R_tilde
            U_1 = (r_u * d_I^{-1/2}) @ V @ V^T * d_I^{+1/2}
            ret = U_2 + alpha * U_1

        W(n_items x n_items) dense 행렬을 실체화하지 않음.
        """
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.asarray(user_ids)

        X_batch_sparse = self.train_matrix_csr[u_ids_np]               # (B, n_items) sparse
        X_np = np.array(X_batch_sparse.todense(), dtype=np.float32)     # (B, n_items)
        X_batch = torch.from_numpy(X_np).to(self.device)                # (B, n_items)

        # --- U_2: linear term  r_u @ R_tilde^T @ R_tilde ---
        norm_adj_t = self.norm_adj.T                                    # (n_items, n_users)
        tmp   = X_np @ norm_adj_t                                       # (B, n_users)
        U_2   = torch.from_numpy(
            np.array(tmp @ self.norm_adj, dtype=np.float32)             # (B, n_items)
        ).to(self.device)

        # --- U_1: ideal low-pass term ---
        # Low-rank property: (X @ V) @ V.T
        proj = X_batch * self.d_mat_i.unsqueeze(0)      # r_u * D_I^{-1/2}  (B, n_items)
        proj = proj @ self.V                             # (B, k)
        proj = proj @ self.V.t()                        # (B, n_items)
        U_1  = proj * self.d_mat_i_inv.unsqueeze(0)     # * D_I^{+1/2}      (B, n_items)

        scores = U_2 + self.alpha * U_1

        if item_ids is not None:
            return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)

        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        scores_full = self.forward(user_ids)
        return scores_full.gather(1, item_ids.unsqueeze(1)).squeeze(1)

    def get_final_item_embeddings(self):
        # V represents item embeddings in the spectral domain
        return self.V

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
