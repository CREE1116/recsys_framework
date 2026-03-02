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
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        
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
                
        self.item_similarity_matrix = sp.csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=self.item_similarity_matrix.shape
        )
            
        self._log("Fitted.")

    def _scipy_sparse_to_torch_sparse(self, sparse_mx):
        """Scipy 희소 행렬을 PyTorch 희소 텐서로 변환합니다."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, users):
        """
        주어진 사용자들에 대한 모든 아이템의 점수를 예측합니다.
        score(u, i) = sum_{j in I(u)} sim(i, j)
        """
        scores = []
        user_ids = users.cpu().numpy()

        # 사용자별로 상호작용 기록을 가져와 점수 계산
        for user_id in user_ids:
            # 사용자가 상호작용한 아이템들의 인덱스
            interacted_items = self.user_item_matrix[user_id].indices
            
            if len(interacted_items) == 0:
                # 상호작용이 없는 경우 0점
                user_scores = torch.zeros(self.n_items)
            else:
                # 사용자가 상호작용한 아이템들과 다른 모든 아이템들 간의 유사도 합
                # item_similarity_matrix: [n_items, n_items]
                # interacted_items: [num_interacted]
                # user_sims: [num_interacted, n_items]
                user_scores = self.item_similarity_matrix[interacted_items, :].sum(axis=0)
                
                # Scipy 결과(matrix)를 numpy array로 변환
                if isinstance(user_scores, np.matrix):
                    user_scores = user_scores.A1
            
            scores.append(torch.FloatTensor(user_scores))
            
        return torch.stack(scores).to(self.device)

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
        """
        ItemKNN 모델은 임베딩이 없으므로, 유사도 행렬을 아이템 표현으로 반환합니다.
        가급적 밀집 행렬(Dense)로 변환하여 반환합니다.
        """
        if self.item_similarity_matrix is None:
            return torch.eye(self.n_items, device=self.device)
        
        if sp.issparse(self.item_similarity_matrix):
            return torch.from_numpy(self.item_similarity_matrix.toarray()).float().to(self.device)
        else:
            return torch.from_numpy(self.item_similarity_matrix).float().to(self.device)

    def calc_loss(self, batch_data):
        """
        이 모델은 학습되지 않으므로 손실은 0입니다.
        """
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return f"ItemKNN(k={self.k})"
