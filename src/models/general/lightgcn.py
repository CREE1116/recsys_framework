import torch
import torch.nn as nn
from ..base_model import BaseModel
from src.loss import BPRLoss

class LightGCN(BaseModel):
    """
    LightGCN: 희소/밀집 인접 행렬을 모두 지원.
    """
    def __init__(self, config, data_loader):
        super(LightGCN, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_layers = self.config['model']['n_layers']

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # 희소/밀집 행렬을 모델의 device로 이동
        self.adj_matrix = self.data_loader.get_interaction_graph().to(self.device)
        
        # 정규화된 인접 행렬 계산
        self.norm_adj_matrix = self._get_normalized_adj_matrix()

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.loss_fn = BPRLoss()
        self._init_weights()
        
        self._final_user_emb = None
        self._final_item_emb = None

    def _get_normalized_adj_matrix(self):
        """[수정] 희소/밀집 행렬에 따라 정규화된 인접 행렬 D^-0.5 * A * D^-0.5 를 계산합니다."""
        A = self.adj_matrix
        
        if A.is_sparse:
            # 희소 행렬 정규화
            A = A.coalesce()
            row_sum = torch.sparse.sum(A, dim=1).to_dense()
            D_inv_sqrt = torch.pow(row_sum, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            
            indices = A.indices()
            values = A.values()
            
            row_degrees = D_inv_sqrt[indices[0]]
            col_degrees = D_inv_sqrt[indices[1]]
            
            new_values = values * row_degrees * col_degrees
            norm_adj = torch.sparse_coo_tensor(indices, new_values, A.shape)
        else:
            # [Optimized] 밀집 행렬 정규화 (Broadcasting 사용)
            # D^-0.5 * A * D^-0.5 = A * d_i * d_j (element-wise)
            D = A.sum(1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            
            # Broadcasting: (N, 1) * (N, N) * (1, N)
            try:
                norm_adj = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)
            except (RuntimeError, MemoryError) as e:
                print(f"[LightGCN] Error in dense graph normalization (OOM?): {e}")
                # Fallback to sparse if dense fails?
                # For now just raise to be caught by global handler
                raise e
            
        return norm_adj

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._final_user_emb, self._final_item_emb = None, None

    def eval(self):
        super().eval()
        self._final_user_emb, self._final_item_emb = None, None

    def _propagate_embeddings(self):
        if not self.training and self._final_user_emb is not None and self._final_item_emb is not None:
            return self._final_user_emb, self._final_item_emb

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        # [수정] 희소/밀집 행렬에 따라 다른 곱셈 연산 사용
        for _ in range(self.n_layers):
            if self.norm_adj_matrix.is_sparse:
                try:
                    all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
                except (RuntimeError, NotImplementedError):
                    # Fallback for MPS which might not support sparse mm
                    all_embeddings = torch.sparse.mm(self.norm_adj_matrix.cpu(), all_embeddings.cpu()).to(self.device)
            else:
                all_embeddings = torch.matmul(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users, self.n_items])

        if not self.training:
            self._final_user_emb = final_user_emb
            self._final_item_emb = final_item_emb

        return final_user_emb, final_item_emb

    def get_embeddings(self):
        return self._propagate_embeddings()

    def get_final_item_embeddings(self):
        """LightGCN의 최종 아이템 임베딩 (Graph-Propagated)을 반환합니다."""
        _, item_embeddings = self.get_embeddings()
        return item_embeddings.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']

        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)

        loss = self.loss_fn(pos_scores, neg_scores)

        return (loss,), None

    def forward(self, users):
        user_embeds, item_embeds = self.get_embeddings()
        user_vecs = user_embeds[users]
        scores = torch.matmul(user_vecs, item_embeds.t())
        return scores
    
    def predict_for_pairs(self, users, items):
        user_embeds, item_embeds = self.get_embeddings()
        
        user_vec = user_embeds[users] # [B, 1, D] or [B, D]
        item_vec = item_embeds[items] # [B, 1, D] or [B, N, D]

        return (user_vec * item_vec).sum(dim=-1)

    def __str__(self):
        return f"LightGCN(embedding_dim={self.embedding_dim}, n_layers={self.n_layers})"