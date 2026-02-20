import torch
import torch.nn as nn
from ..base_model import BaseModel
from src.loss import BPRLoss, MSELoss, SampledSoftmaxLoss

class MF(BaseModel):
    """
    Matrix Factorization 모델 (Pairwise BPR)
    """
    def __init__(self, config, data_loader):
        super(MF, self).__init__(config, data_loader)

        self.embedding_dim = config['model']['embedding_dim']
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.loss_fn = BPRLoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def predict_for_pairs(self, users, items):
        user_embeds = self.user_embedding(users)  # [B, D]
        item_embeds = self.item_embedding(items)  # [B, D] or [B, N, D]
        
        return torch.sum(user_embeds * item_embeds, dim=-1)

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']

        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
    
        loss = self.loss_fn(pos_scores, neg_scores)
        
        return (loss,), None

    def forward(self, users):
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding.weight
        scores = torch.matmul(user_embeds, item_embeds.transpose(0, 1))
        return scores

    def get_final_item_embeddings(self):
        """MF의 최종 아이템 임베딩은 기본 임베딩과 동일합니다."""
        return self.item_embedding.weight.detach()

    def get_item_embeddings(self):
        return self.item_embedding.weight
