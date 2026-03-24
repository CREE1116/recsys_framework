import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager


class GF_CF(BaseModel):
    """
    GF-CF: Graph Filter based Collaborative Filtering (CIKM 2021)
    Exact Implementation with materialized weight matrix.
    
    W = R_tilde^T @ R_tilde + alpha * (V @ V^T)
    where R_tilde is the symmetrically normalized adjacency matrix.
    """
    def __init__(self, config, data_loader):
        super(GF_CF, self).__init__(config, data_loader)
        
        # Hyperparameters
        k = self.config['model'].get('k', 256)
        self.k = k[0] if isinstance(k, list) else k
        self.alpha = self.config['model'].get('alpha', 0.5)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # SVD Manager initialization
        self.svd_manager = SVDCacheManager(device=self.device.type)
        
        self.register_buffer('W', torch.empty(0))
        self.train_matrix_csr = None
        
        self._log(f"Initialized (k={self.k}, alpha={self.alpha})")

        # Cache manager 등록
        self.register_cache_manager('svd', self.svd_manager)

    def _normalize_matrix(self, R):
        """Symmetric Normalization: D_u^-0.5 * R * D_i^-0.5"""
        user_sums = np.array(R.sum(axis=1)).flatten()
        item_sums = np.array(R.sum(axis=0)).flatten()
        
        user_inv_sqrt = np.zeros_like(user_sums)
        u_mask = user_sums > 0
        user_inv_sqrt[u_mask] = np.power(user_sums[u_mask], -0.5)
        
        item_inv_sqrt = np.zeros_like(item_sums)
        i_mask = item_sums > 0
        item_inv_sqrt[i_mask] = np.power(item_sums[i_mask], -0.5)
        
        R_norm = sp.diags(user_inv_sqrt) @ R @ sp.diags(item_inv_sqrt)
        return R_norm

    def fit(self, data_loader):
        """Build exact weight matrix W."""
        self._log("Building interaction matrix...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        R = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        self.train_matrix_csr = R
        
        # 1. Normalize
        self._log("Normalizing matrix (Symmetric)...")
        R_tilde = self._normalize_matrix(R)
        
        # 2. SVD via SVDCacheManager
        dataset_name = self.config.get('dataset_name', 'unknown')
        self._log(f"Performing SVD (k={self.k})...")
        u, s, v, _ = self.svd_manager.get_svd(R_tilde, k=self.k, dataset_name=dataset_name)
        
        # Ensure exact K is used (safety truncation)
        V = v[:, :self.k].to(self.device).float() # (n_items, k)
        
        # 3. Linear term P = R_tilde^T @ R_tilde
        self._log("Computing Linear term (Gram matrix)...")
        # Produced matrix is sparse (n_items x n_items)
        P_sparse = R_tilde.T @ R_tilde
        
        # 4. Low-pass term L = V @ V^T
        self._log("Computing Low-pass term (V @ V^T)...")
        L_dense = torch.mm(V, V.t()) # (n_items, n_items)
        
        # 5. Final W = P + alpha * L
        self._log("Assembling final weight matrix W...")
        # Convert sparse P to dense on device
        P_dense = torch.from_numpy(P_sparse.toarray()).float().to(self.device)
        self.W = P_dense + self.alpha * L_dense
        
        del P_sparse, P_dense, L_dense, R_tilde
        self._log(f"Fitted. W shape: {self.W.shape}")

    def forward(self, user_ids, item_ids=None):
        """Predict scores using W."""
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids

        X_batch_sparse = self.train_matrix_csr[u_ids_np]
        X_batch = torch.from_numpy(X_batch_sparse.toarray()).float().to(self.device)
        
        scores = torch.matmul(X_batch, self.W)
        
        if item_ids is not None:
            # Pairwise prediction
            return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)
            
        return scores

    def get_final_item_embeddings(self):
        # W itself can be seen as item similarity/embeddings
        return self.W

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
