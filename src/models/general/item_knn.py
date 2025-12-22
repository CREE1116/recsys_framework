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
        
        print(f"ItemKNN model initialized with k={self.k}.")

    def fit(self, data_loader):
        """
        ItemKNN 모델의 유사도 행렬을 계산합니다.
        이 메소드는 main.py에서 학습이 필요 없는 모델에 대해 호출됩니다.
        """
        # data_loader는 self.data_loader와 동일한 인스턴스
        print("Building interaction matrix for ItemKNN...")
        self.user_item_matrix = sp.csr_matrix(
            (np.ones(len(data_loader.train_df)), 
             (data_loader.train_df['user_id'], data_loader.train_df['item_id'])),
            shape=(data_loader.n_users, data_loader.n_items)
        )
        
        print(f"Calculating item-item similarity matrix using {self.similarity_metric}...")
        if self.similarity_metric == 'jaccard':
            # Manual Jaccard Calculation using Sparse Matrix Multiplication
            # Much faster than sklearn pairwise_distances for sparse data
            
            # 1. Ensure binary matrix (Items x Users)
            # self.user_item_matrix is (Users x Items). We need (Items x Users) for item-item similarity.
            # actually similarity between columns of user_item_matrix.
            # Let X = user_item_matrix (Users x Items).
            # We want similarity between items i and j. Be careful with orientation.
            # sklearn cosine_similarity(X.T) computes sim between rows of X.T (which are items).
            
            X = self.user_item_matrix.T # (Items x Users)
            X.data = np.ones_like(X.data) # Ensure binary
            
            # 2. Intersection: X @ X.T -> (Items x Items) element (i, j) is number of users who interacted with both i and j.
            print("Calculating intersection (X @ X.T)...")
            intersection = X @ X.T 

            # 3. Union: |A| + |B| - Intersection
            # |A| is number of users for item A (row sum of X)
            item_counts = np.array(X.sum(axis=1)).flatten() # (n_items,)
            
            # We need matrix where M[i, j] = count[i] + count[j]
            # Use broadcasting: (N, 1) + (1, N) -> (N, N)
            print("Calculating union...")
            count_matrix = item_counts[:, None] + item_counts[None, :]
            
            # interaction is sparse, count_matrix is dense.
            # But we only need values where union > 0.
            # However, Jaccard is dense if we want full matrix.
            # But usually we only care about non-zero intersections for KNN?
            # Actually weak generalization: 0 intersection -> 0 similarity.
            # So we can just operate on sparse structure of intersection?
            # BUT intersection is likely denser than X.
            # 3000 items -> 9M entries. Dense is fine (72MB).
            
            intersection_dense = intersection.toarray()
            union_dense = count_matrix - intersection_dense
            
            # Avoid division by zero
            valid_mask = union_dense > 0
            self.item_similarity_matrix = np.zeros_like(intersection_dense, dtype=np.float32)
            self.item_similarity_matrix[valid_mask] = intersection_dense[valid_mask] / union_dense[valid_mask]
            
            
        elif self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
        print("ItemKNN model fitted successfully.")

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

    def get_embeddings(self):
        """
        ItemKNN 모델은 임베딩이 없으므로, ILD 계산을 위해 None을 반환합니다.
        evaluation.py에서 이를 자동으로 건너뜁니다.
        """
        return None, None

    def get_final_item_embeddings(self):
        """
        ItemKNN 모델은 임베딩이 없으므로, 시각화를 위한 placeholder로 단위 행렬을 반환합니다.
        그러나 get_embeddings()가 None을 반환하므로 사실상 사용되지 않습니다.
        """
        return torch.eye(self.n_items, device=self.device).detach()

    def calc_loss(self, batch_data):
        """
        이 모델은 학습되지 않으므로 손실은 0입니다.
        """
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return f"ItemKNN(k={self.k})"
