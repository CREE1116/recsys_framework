import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
class MACR(BaseModel):
    def __init__(self, config, data_loader):
        super(MACR, self).__init__(config, data_loader)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        self.embedding_dim = self.config['model'].get('embedding_dim', 64)
        
        # Hyperparameters
        self.c = self.config['model'].get('c', 30.0)  # 논문에서 30-40 추천
        self.alpha = self.config['model'].get('alpha', 1e-3)
        self.beta = self.config['model'].get('beta', 1e-3)
        
        # Main branch (User-Item Matching)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Item branch (Popularity) - 별도 임베딩 또는 MLP
        self.item_pop_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_pop_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            # Sigmoid는 나중에 적용
        )
        
        # User branch (Activity) - 별도 임베딩 또는 MLP
        self.user_act_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_act_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            # Sigmoid는 나중에 적용
        )
        
        self._init_weights()

    def _init_weights(self):
        for module in [self.user_embedding, self.item_embedding, 
                       self.item_pop_embedding, self.user_act_embedding]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, users, items):
        """
        Training forward:
        y_hat = y_ui * sigma(y_i) * sigma(y_u)
        """
        # Main matching score
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        y_ui = (u_emb * i_emb).sum(dim=-1)  # [B]
        
        # Item popularity (별도 임베딩)
        i_pop_emb = self.item_pop_embedding(items)
        y_i = self.item_pop_mlp(i_pop_emb).squeeze()  # [B]
        
        # User activity (별도 임베딩)
        u_act_emb = self.user_act_embedding(users)
        y_u = self.user_act_mlp(u_act_emb).squeeze()  # [B]
        
        return y_ui, y_u, y_i

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        items = batch_data['item_id'].squeeze(-1)
        labels = batch_data['label'].float().squeeze(-1)
        
        y_ui, y_u, y_i = self.forward(users, items)
        
        # Main loss: y_hat = y_ui * sigma(y_i) * sigma(y_u)
        # Logit space: log(y_hat) = y_ui + log(sigma(y_i)) + log(sigma(y_u))
        #             ≈ y_ui + y_i + y_u (근사)
        # 하지만 논문은 곱셈 형태를 유지하므로:
        y_hat = y_ui * torch.sigmoid(y_i) * torch.sigmoid(y_u)
        loss_main = F.binary_cross_entropy_with_logits(y_hat, labels)
        
        # Auxiliary losses (Multi-task)
        loss_i = F.binary_cross_entropy_with_logits(y_i, labels)
        loss_u = F.binary_cross_entropy_with_logits(y_u, labels)
        
        params_to_log = {
            'main_loss': loss_main.item(),
            'item_loss': loss_i.item(),
            'user_loss': loss_u.item()
        }
        
        return (loss_main , self.alpha * loss_i , self.beta * loss_u,), params_to_log

    def predict_for_pairs(self, user_ids, item_ids):
        """
        Inference (Counterfactual):
        score = y_ui * sigma(y_u) * [sigma(y_i) - c * sigma(y_i*)]
        
        논문의 단순화: score ≈ y_ui - c * y_i
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        y_ui = (u_emb * i_emb).sum(dim=-1)
        
        # Popularity (counterfactual에서 제거)
        i_pop_emb = self.item_pop_embedding(item_ids)
        y_i = self.item_pop_mlp(i_pop_emb).squeeze()
        
        # 논문의 핵심: TIE (Total Indirect Effect) 제거
        scores = y_ui - self.c * torch.sigmoid(y_i)
        
        return scores

    def get_final_item_embeddings(self):
        return self.item_embedding.weight