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
        
        print("Calculating item-item similarity matrix...")
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        
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
