import torch
import torch.nn as nn
from ..base_model import BaseModel
from src.loss import NormalizedSampledSoftmaxLoss


class LightGCN_Sampled(BaseModel):
    """
    LightGCN + Sampled Softmax Loss
    GCN 구조는 동일, Loss만 NormalizedSampledSoftmaxLoss로 변경
    """
    def __init__(self, config, data_loader):
        super(LightGCN_Sampled, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_layers = self.config['model']['n_layers']
        
        # Sampled Softmax 관련 설정
        self.temperature = self.config['model'].get('temperature', 0.1)
        self.use_zscore = self.config['model'].get('use_zscore', False)
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.is_explicit = self.num_negatives > 0

        self.num_users = self.data_loader.n_users
        self.num_items = self.data_loader.n_items
        
        # 희소/밀집 행렬을 모델의 device로 이동
        self.adj_matrix = self.data_loader.get_interaction_graph().to(self.device)
        
        # 정규화된 인접 행렬 계산
        self.norm_adj_matrix = self._get_normalized_adj_matrix()

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # Sampled Softmax Loss
        self.loss_fn = NormalizedSampledSoftmaxLoss(
            self.num_items, 
            temperature=self.temperature, 
            use_zscore=self.use_zscore
        )
        
        self._init_weights()
        
        self._final_user_emb = None
        self._final_item_emb = None

    def _get_normalized_adj_matrix(self):
        """희소/밀집 행렬에 따라 정규화된 인접 행렬 D^-0.5 * A * D^-0.5 를 계산합니다."""
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
            # 밀집 행렬 정규화
            D = A.sum(1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            
            norm_adj = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)
            
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

        for _ in range(self.n_layers):
            if self.norm_adj_matrix.is_sparse:
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            else:
                all_embeddings = torch.matmul(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])

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
        user_embeds, item_embeds = self.get_embeddings()

        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)

        user_vec = user_embeds[users]  # [B, D]
        pos_vec = item_embeds[pos_items]  # [B, D]
        
        if self.is_explicit:
            # Explicit Negative Sampling
            neg_items = batch_data['neg_item_id']  # [B, N]
            
            # Pos Scores: [B, 1]
            pos_scores = (user_vec * pos_vec).sum(dim=-1, keepdim=True)
            
            # Neg Scores: [B, N]
            B, N = neg_items.size()
            neg_vec = item_embeds[neg_items.view(-1)].view(B, N, -1)  # [B, N, D]
            neg_scores = torch.einsum('bd,bnd->bn', user_vec, neg_vec)
            
            # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
        else:
            # In-Batch Negatives: [B, B]
            scores = torch.matmul(user_vec, pos_vec.t())
        
        loss = self.loss_fn(scores, is_explicit=self.is_explicit)

        return (loss,), None

    def forward(self, users):
        user_embeds, item_embeds = self.get_embeddings()
        user_vecs = user_embeds[users]
        scores = torch.matmul(user_vecs, item_embeds.t())
        return scores
    
    def predict_for_pairs(self, users, items):
        user_embeds, item_embeds = self.get_embeddings()
        
        user_vec = user_embeds[users]
        item_vec = item_embeds[items]

        if item_vec.dim() == 2:
            scores = torch.sum(user_vec * item_vec, dim=1)
        elif item_vec.dim() == 3:
            scores = torch.einsum('bd,bnd->bn', user_vec, item_vec)
        else:
            raise ValueError(f"Invalid item tensor shape: {item_vec.shape}")
            
        return scores

    def __str__(self):
        return f"LightGCN_Sampled(embedding_dim={self.embedding_dim}, n_layers={self.n_layers}, temperature={self.temperature})"
