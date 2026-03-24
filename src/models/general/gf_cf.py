import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager


class GF_CF(BaseModel):
    """
    GF-CF: Graph Filter based Collaborative Filtering (CIKM 2021)
    Optimized for both Speed (Inference) and Memory (Large Datasets).

    최적화 전략:
    1. Adaptive Materialization: 아이템 수(N)가 적으면(예: < 15,000) W를 직접 생성하여 인퍼런스 속도 최적화.
    2. GPU Sparse (CUDA): N이 크면 W를 생성하지 않고 GPU Sparse 연산을 활용해 메모리 절약.
    3. Efficient Pair Prediction: 쌍 예측 시 불필요한 전체 아이템 점수 계산 제거.
    4. MPS Compatibility: MPS 환경에서는 Sparse GPU 연산 대신 최적화된 CPU/Dense 연산 수행.
    """

    def __init__(self, config, data_loader):
        super(GF_CF, self).__init__(config, data_loader)

        k = self.config['model'].get('k', 256)
        self.k = k[0] if isinstance(k, list) else k
        self.alpha = self.config['model'].get('alpha', 0.3)
        self.materialize_threshold = self.config['model'].get('materialize_threshold', 20000)

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        self.svd_manager = SVDCacheManager(device=self.device.type)
        self.register_cache_manager('svd', self.svd_manager)

        # 상태 변수
        self.train_matrix_csr = None
        self.W                = None # Materialized weight matrix (Optional)
        self.norm_adj_torch   = None # GPU Sparse R_tilde
        self.norm_adj_t_torch = None # GPU Sparse R_tilde^T
        self.V                = None 
        self.d_mat_i          = None
        self.d_mat_i_inv      = None

        self._log(f"Initialized (k={self.k}, alpha={self.alpha}, items={self.n_items})")

    def _to_torch_sparse(self, sp_mat):
        """Convert scipy sparse to torch sparse (CSR preferred for mm)"""
        if sp_mat is None: return None
        sp_mat = sp_mat.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(sp_mat.indptr).to(torch.int64),
            torch.from_numpy(sp_mat.indices).to(torch.int64),
            torch.from_numpy(sp_mat.data).to(torch.float32),
            size=sp_mat.shape,
            device=self.device
        )

    def fit(self, data_loader):
        self._log("Building interaction matrix...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.n_users, self.n_items))
        self.train_matrix_csr = R

        # 1. Normalization
        rowsum = np.array(R.sum(axis=1)).flatten()
        d_inv_row = np.power(rowsum, -0.5)
        d_inv_row[np.isinf(d_inv_row)] = 0.
        
        colsum = np.array(R.sum(axis=0)).flatten()
        d_inv_col = np.power(colsum, -0.5)
        d_inv_col[np.isinf(d_inv_col)] = 0.

        norm_adj = sp.diags(d_inv_row) @ R @ sp.diags(d_inv_col)
        norm_adj = norm_adj.tocsc()

        self.d_mat_i     = torch.from_numpy(d_inv_col.astype(np.float32)).to(self.device)
        self.d_mat_i_inv = torch.from_numpy(np.where(d_inv_col > 0, 1.0 / d_inv_col, 0.0).astype(np.float32)).to(self.device)

        # 2. SVD
        dataset_name = self.config.get('dataset_name', 'unknown')
        u, s, v, _ = self.svd_manager.get_svd(norm_adj, k=self.k, dataset_name=dataset_name)
        self.V = v[:, :self.k].to(self.device).float()

        # 3. Adaptive Strategy
        if self.n_items < self.materialize_threshold:
            self._log(f"Materializing W matrix (N={self.n_items} < {self.materialize_threshold})...")
            # Linear term: P = R_tilde^T @ R_tilde
            P = (norm_adj.T @ norm_adj).toarray().astype(np.float32)
            P = torch.from_numpy(P).to(self.device)
            # Low-pass term: L = D_i^-0.5 V V^T D_i^0.5
            L = self.V @ self.V.t()
            L = self.d_mat_i.unsqueeze(1) * L * self.d_mat_i_inv.unsqueeze(0)
            self.W = P + self.alpha * L
            self._log("Materialization complete. Inference will be fast.")
        else:
            self._log(f"Memory-efficient mode enabled (N={self.n_items}).")
            if self.device.type == 'cuda':
                self._log("Moving sparse matrices to GPU...")
                self.norm_adj_torch = self._to_torch_sparse(norm_adj)
                self.norm_adj_t_torch = self._to_torch_sparse(norm_adj.T)
            else:
                self._log("MPS/CPU detected. Keeping sparse matrices in Scipy.")

    def forward(self, user_ids, item_ids=None):
        if self.W is not None:
            X_batch = torch.from_numpy(self.train_matrix_csr[user_ids.cpu().numpy()].toarray()).to(self.device)
            scores = X_batch @ self.W
            if item_ids is not None:
                return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)
            return scores

        # On-the-fly optimized path
        u_ids_np = user_ids.cpu().numpy()
        X_batch_sparse = self.train_matrix_csr[u_ids_np]
        X_batch_dense = torch.from_numpy(X_batch_sparse.toarray()).to(self.device)

        # --- Linear term U_2 ---
        if self.norm_adj_torch is not None:
            # step1: X @ R_tilde^T = (B, n_items) @ (n_items, n_users)
            # torch.sparse.mm(sparse, dense_t).t() -> (n_users, n_items) @ (n_items, B) -> (n_users, B).t() -> (B, n_users)
            tmp = torch.sparse.mm(self.norm_adj_torch, X_batch_dense.t()).t()
            # step2: tmp @ R_tilde = (B, n_users) @ (n_users, n_items)
            # torch.sparse.mm(sparse, dense_t).t() -> (n_items, n_users) @ (n_users, B) -> (n_items, B).t() -> (B, n_items)
            U_2 = torch.sparse.mm(self.norm_adj_t_torch, tmp.t()).t()
        else:
            # MPS/CPU Scipy Fallback
            tmp = X_batch_sparse @ self.norm_adj_t_cpu
            U_2 = torch.from_numpy((tmp @ self.norm_adj_cpu).toarray()).to(self.device)

        # --- Low-pass term U_1 ---
        proj = (X_batch_dense * self.d_mat_i.unsqueeze(0)) @ self.V
        U_1  = (proj @ self.V.t()) * self.d_mat_i_inv.unsqueeze(0)

        scores = U_2 + self.alpha * U_1
        if item_ids is not None:
            return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # 쌍 예측 시 BxN 행렬 생성을 방지하여 메모리/속도 최적화
        if self.W is not None or len(user_ids) > 5000:
            # 배치 크기가 크거나 W가 있으면 forward가 더 유리할 수 있음
            return self.forward(user_ids, item_ids)

        u_ids_np = user_ids.cpu().numpy()
        X_batch_sparse = self.train_matrix_csr[u_ids_np]
        X_batch_dense = torch.from_numpy(X_batch_sparse.toarray()).to(self.device)
        
        # Low-pass part: O(B * K)
        proj = (X_batch_dense * self.d_mat_i.unsqueeze(0)) @ self.V # (B, K)
        V_items = self.V[item_ids] # (B, K)
        scores_lp = (proj * V_items).sum(dim=1) * self.d_mat_i_inv[item_ids]

        # Linear part: O(B * nnz_row)
        tmp = X_batch_sparse @ self.norm_adj_t_cpu
        scores_lin = []
        for b in range(len(u_ids_np)):
            scores_lin.append(tmp[b] @ self.norm_adj_cpu[:, item_ids[b].item()])
        scores_lin = torch.tensor(scores_lin, device=self.device).float()

        return scores_lin + self.alpha * scores_lp

    def get_final_item_embeddings(self):
        return self.V

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
