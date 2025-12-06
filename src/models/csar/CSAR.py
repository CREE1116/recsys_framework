import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss 

from ..base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        self.soft_relu = self.config['model'].get('soft_relu', False)
        # Standard Deviation Scaling Factor (Beta) for Power-Scaled Norm
        self.std_power = self.config['model'].get('std_power', 0.2)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight

        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)

        scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)

        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        items = batch_data['item_id']

        preds = self.forward(users) 
        
        # # Power-Scaled Normalization
        # # preds.std() ** beta reduces the aggressive scaling of high-variance dense data
        # std_scaling = preds.std().pow(self.std_power) + 1e-9
        # preds = (preds - preds.mean()) / std_scaling
        
        # Global Softmax Cross Entropy
        loss = F.cross_entropy(preds, items) 

        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")

        params_to_log = {'scale': self.attention_layer.scale.item()}

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
