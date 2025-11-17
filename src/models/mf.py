import torch
import torch.nn as nn
from .base_model import BaseModel
from src.loss import BPRLoss, MSELoss, InfoNCELoss

class MF(BaseModel):
    """
    Matrix Factorization 모델 (Pairwise BPR/InfoNCE Loss 전용)
    """
    def __init__(self, config, data_loader):
        super(MF, self).__init__(config, data_loader)

        self.embedding_dim = config['model']['embedding_dim']
        self.num_negatives = config['train'].get('num_negatives', 1)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)

        if self.num_negatives == 1:
            self.loss_fn = BPRLoss()
            self.loss_name = 'bpr_loss'
        else:
            self.loss_fn = InfoNCELoss(config)
            self.loss_name = 'infonce_loss'

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, items):
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding(items)
        
        output = torch.sum(user_embeds * item_embeds, dim=-1)
        return output

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id'] # Trainer에서 이미 처리됨

        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        
        # InfoNCE Loss의 경우 neg_scores가 (batch_size, num_negatives) 형태여야 함
        # BPR Loss의 경우 neg_scores가 (batch_size, 1) 형태여야 함
        if self.num_negatives == 1 and neg_scores.dim() > 1: # BPR인데 하드 네거티브가 여러개일 경우 첫번째만 사용
            neg_scores = neg_scores[:, 0].unsqueeze(1)
        elif self.num_negatives > 1 and neg_scores.dim() == 1: # InfoNCE인데 neg_scores가 1차원일 경우 (batch_size, 1)로 변경
            neg_scores = neg_scores.unsqueeze(1)

        loss = self.loss_fn(pos_scores, neg_scores)
       
        
        return (loss,), None

    def predict(self, users):
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding.weight
        scores = torch.matmul(user_embeds, item_embeds.transpose(0, 1))
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        scores = torch.sum(user_embeds * item_embeds, dim=1)
        return scores

    def get_item_embeddings(self):
        return self.item_embedding.weight
