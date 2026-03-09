import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from ..base_model import BaseModel

class ItemKNN(BaseModel):
    """
    Item-based k-NN Collaborative Filtering 모델.
    이 모델은 학습되지 않으며, __init__ 시점에서 유사도 행렬을 미리 계산합니다.
    """
    def __init__(self, config, data_loader):
        super(ItemKNN, self).__init__(config, data_loader)
        
        self.k = self.config['model'].get('k', 50)
        self.similarity_metric = self.config['model'].get('similarity_metric', 'cosine')
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # main.py에서 fit()을 명시적으로 호출하므로, __init__에서는 데이터 로드만 준비
        # Sparse buffers for interaction and similarity
        self.register_buffer('ui_indices', torch.empty(2, 0, dtype=torch.long))
        self.register_buffer('ui_values', torch.empty(0))
        self.register_buffer('sim_indices', torch.empty(2, 0, dtype=torch.long))
        self.register_buffer('sim_values', torch.empty(0))
        
        self.ui_shape = (self.n_users, self.n_items)
        self.sim_shape = (self.n_items, self.n_items)
        
        self._log(f"Initialized (k={self.k})")

    def fit(self, data_loader):
        """
        ItemKNN 모델의 유사도 행렬을 계산합니다.
        이 메소드는 main.py에서 학습이 필요 없는 모델에 대해 호출됩니다.
        """
        # data_loader는 self.data_loader와 동일한 인스턴스
        self._log("Building interaction matrix...")
        self.user_item_matrix = sp.csr_matrix(
            (np.ones(len(data_loader.train_df)), 
             (data_loader.train_df['user_id'], data_loader.train_df['item_id'])),
            shape=(data_loader.n_users, data_loader.n_items)
        )
        
        self._log(f"Calculating similarity ({self.similarity_metric})...")
        if self.similarity_metric == 'jaccard':
            # Manual Jaccard Calculation using Sparse Matrix Multiplication
            X = self.user_item_matrix.T  # (Items x Users)
            X.data = np.ones_like(X.data)  # Ensure binary
            
            self._log("Calculating intersection (X @ X.T)...")
            intersection = X @ X.T 

            item_counts = np.array(X.sum(axis=1)).flatten()
            self._log("Calculating union...")
            count_matrix = item_counts[:, None] + item_counts[None, :]
            
            intersection_dense = intersection.toarray()
            union_dense = count_matrix - intersection_dense
            
            valid_mask = union_dense > 0
            self.item_similarity_matrix = np.zeros_like(intersection_dense, dtype=np.float32)
            self.item_similarity_matrix[valid_mask] = intersection_dense[valid_mask] / union_dense[valid_mask]
            
        elif self.similarity_metric == 'cosine':
            try:
                # Try sparse output if possible (requires scikit-learn >= 0.24 approx?)
                # dense_output=False is not always supported by cosine_similarity in older versions?
                # It's better to check matrix size.
                if self.n_items > 10000:
                    self._log(f"Warning: n_items={self.n_items} is large. Cosine similarity might OOM.")
                
                self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T, dense_output=False)
            except TypeError:
                # Fallback if dense_output param not supported
                self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            except (MemoryError, RuntimeError) as e:
                self._log(f"OOM during cosine_similarity: {e}")
                raise e
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
        # --- Top-K Pruning ---
        self._log(f"Top-{self.k} pruning...")
        
        # Convert to CSR if not already (sklean cosine_similarity might return dense or csr)
        if not sp.issparse(self.item_similarity_matrix):
            self.item_similarity_matrix = sp.csr_matrix(self.item_similarity_matrix)
            
        # Row-wise top-k selection
        n_items = self.item_similarity_matrix.shape[0]
        
        # Efficient row-wise top-k for CSR matrix
        # We can iterate or use specialized methods. For 40k items, iteration is okay-ish (seconds).
        # Better: use LIL for modifying or rebuild CSR data.
        
        # Strategy: Iterate rows, find k-th largest, mask lower.
        # But doing this in pure python loop for 38k rows is slow.
        # Alternative: Use numpy on indptr/indices/data directly?
        
        # Let's use a robust approach:
        rows = []
        cols = []
        data = []
        
        # To avoid slow python loops, we can try to do it in batches or parallel, 
        # but simple loop might be acceptable if N is not huge (38k is borderline).
        # Let's try a vectorized approach if possible? 
        # No, irregular number of non-zeros per row.
        
        # Let's stick to a clean loop with optimizations (e.g. numpy sorting per row)
        # Using LIL might be faster for specific access but CSR + slicing is standard.
        
        # Wait, sklearn has no direct 'kneighbors_graph' for precomputed similarity?
        # Actually, let's just loop. 38,000 items is fine for a linear scan of rows.
        
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for i in tqdm(range(n_items), desc="Pruning KNN"):
            row_start = self.item_similarity_matrix.indptr[i]
            row_end = self.item_similarity_matrix.indptr[i+1]
            
            row_data = self.item_similarity_matrix.data[row_start:row_end]
            row_indices = self.item_similarity_matrix.indices[row_start:row_end]
            
            if len(row_data) <= self.k:
                # Keep all
                new_data.extend(row_data)
                new_indices.extend(row_indices)
                new_indptr.append(new_indptr[-1] + len(row_data))
            else:
                # Find top-k
                # argsort is ascending, so we take last k
                # but we want largest values.
                # argpartition is faster O(n) vs O(n log n)
                
                # Careful: zero values in sparse matrix? Usually not stored.
                # We want top-k absolute values? Similarity is usually positive.
                
                idx = np.argpartition(row_data, -self.k)[-self.k:]
                
                # Filter
                top_data = row_data[idx]
                top_indices = row_indices[idx]
                
                new_data.extend(top_data)
                new_indices.extend(top_indices)
                new_indptr.append(new_indptr[-1] + self.k)
                
        # Convert to Torch Sparse and register as buffers
        self.item_similarity_matrix = self.item_similarity_matrix.tocoo()
        self.register_buffer('sim_indices', torch.LongTensor([self.item_similarity_matrix.row, self.item_similarity_matrix.col]))
        self.register_buffer('sim_values', torch.FloatTensor(self.item_similarity_matrix.data))
        
        ui_coo = self.user_item_matrix.tocoo()
        self.register_buffer('ui_indices', torch.LongTensor([ui_coo.row, ui_coo.col]))
        self.register_buffer('ui_values', torch.FloatTensor(ui_coo.data))
        
        self._log("Fitted and stored as Torch Sparse buffers.")

    def _scipy_sparse_to_torch_sparse(self, sparse_mx):
        """Scipy 희소 행렬을 PyTorch 희소 텐서로 변환합니다."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, users):
        """
        [Optimization] GPU-accelerated Sparse ItemKNN Score Calculation
        score = X @ S (where X is user history multi-hot, S is item-item similarity)
        """
        # Build user history batch (Dense for small batches, or Sparse if huge)
        batch_size = users.size(0)
        user_input = torch.zeros(batch_size, self.n_items, device=self.device)
        
        # User history is stored in buffers
        # We need a way to efficiently get batch user history on GPU
        # If we use sparse tensors: batch_users = self.user_item_matrix[users]
        # Torch sparse indexing isn't great. Let's use the loader's history if needed or dense batch.
        
        users_np = users.cpu().numpy()
        for i, u_id in enumerate(users_np):
            hist = list(self.data_loader.train_user_history.get(int(u_id), []))
            if hist:
                user_input[i, hist] = 1.0
        
        # S = similarity matrix (Sparse)
        S = torch.sparse_coo_tensor(self.sim_indices, self.sim_values, self.sim_shape, device=self.device)
        
        # Score = X @ S
        # torch.sparse.mm(sparse_A, dense_B) -> we need X @ S which is dense @ sparse
        # We can use (S^T @ X^T)^T
        if self.device.type == 'mps':
            # MPS does not support torch.sparse.mm; fall back to dense matmul
            scores = torch.mm(user_input, S.to_dense())
        else:
            scores = torch.sparse.mm(S.t(), user_input.t()).t()
            
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """
        주어진 (사용자, 아이템) 쌍에 대한 점수를 계산합니다.
        """
        scores = []
        user_ids_np = user_ids.cpu().numpy()
        item_ids_np = item_ids.cpu().numpy()

        for user_id, item_id in zip(user_ids_np, item_ids_np):
            interacted_items = self.user_item_matrix[user_id].indices
            
            if len(interacted_items) == 0:
                score = 0.0
            else:
                # 특정 아이템(item_id)과 상호작용한 아이템들 간의 유사도 합
                sim_scores = self.item_similarity_matrix[item_id, interacted_items].sum()
                score = sim_scores
            
            scores.append(score)
            
        return torch.FloatTensor(scores).to(self.device)

    def get_final_item_embeddings(self):
        """Item-Item similarity is used as item 'embeddings'."""
        if self.sim_values.numel() == 0:
            return torch.eye(self.n_items, device=self.device)
        
        # Only toarray if items are small
        if self.n_items < 10000:
            S = torch.sparse_coo_tensor(self.sim_indices, self.sim_values, self.sim_shape, device=self.device)
            return S.to_dense()
        else:
            self._log(f"Warning: n_items={self.n_items} too large for dense embeddings. Returning None.")
            return None

    def calc_loss(self, batch_data):
        """
        이 모델은 학습되지 않으므로 손실은 0입니다.
        """
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return f"ItemKNN(k={self.k})"
