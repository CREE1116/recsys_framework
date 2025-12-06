import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, embedding_dim):
        super(PrototypeLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
    def forward(self, embeddings):
        # Calculate similarity (dot product) -> Similarity Distribution
        # ProtoMF Paper uses cosine similarity usually, or simple dot product + Softmax
        scores = torch.matmul(embeddings, self.prototypes.T)
        distribution = F.softmax(scores, dim=-1)
        
        # Reconstruct embedding based on prototypes
        reconstructed = torch.matmul(distribution, self.prototypes)
        return reconstructed, distribution
    
    def orthogonal_loss(self):
        # Encourage prototypes to be distinct
        start_norm = F.normalize(self.prototypes, dim=1)
        sim_matrix = torch.matmul(start_norm, start_norm.T)
        I = torch.eye(self.prototypes.size(0), device=sim_matrix.device)
        return torch.norm(sim_matrix - I)

class ProtoMF(BaseModel):
    """
    ProtoMF: Prototype-based Matrix Factorization
    - Uses M user prototypes and N item prototypes.
    - Represents users/items as linear combinations of their respective prototypes.
    - Offers explainability via prototype association.
    """
    def __init__(self, config, data_loader):
        super(ProtoMF, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_prototypes = self.config['model']['num_prototypes'] # K
        self.lambda_orth = self.config['model'].get('lambda_orth', 0.01)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # Separate prototypes for Users and Items (RecSys '22 standard approach)
        self.user_proto_layer = PrototypeLayer(self.num_prototypes, self.embedding_dim)
        self.item_proto_layer = PrototypeLayer(self.num_prototypes, self.embedding_dim)
        
        self._init_weights()
        self.loss_fn = nn.LogSigmoid()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        u_emb_raw = self.user_embedding(users)
        i_emb_raw_all = self.item_embedding.weight
        
        u_emb_proto, _ = self.user_proto_layer(u_emb_raw)
        i_emb_proto_all, _ = self.item_proto_layer(i_emb_raw_all)
        
        scores = torch.matmul(u_emb_proto, i_emb_proto_all.T)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb_raw = self.user_embedding(user_ids)
        i_emb_raw = self.item_embedding(item_ids)
        
        u_emb_proto, _ = self.user_proto_layer(u_emb_raw)
        i_emb_proto, _ = self.item_proto_layer(i_emb_raw)
        
        return (u_emb_proto * i_emb_proto).sum(dim=-1)

    def get_final_item_embeddings(self):
        i_emb_raw_all = self.item_embedding.weight
        i_emb_proto_all, _ = self.item_proto_layer(i_emb_raw_all)
        return i_emb_proto_all.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        # Forward inputs
        u_emb, u_dist = self.user_proto_layer(self.user_embedding(users))
        pos_i_emb, pos_i_dist = self.item_proto_layer(self.item_embedding(pos_items))
        neg_i_emb, neg_i_dist = self.item_proto_layer(self.item_embedding(neg_items))
        
        # BPR Loss
        pos_scores = (u_emb * pos_i_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_i_emb).sum(dim=-1)
        bpr_loss = -self.loss_fn(pos_scores - neg_scores).mean()
        
        # Orthogonality Regulation (Prototypes diversity)
        orth_loss = self.user_proto_layer.orthogonal_loss() + self.item_proto_layer.orthogonal_loss()
        
        params_to_log = {
            'bpr_loss': bpr_loss.item(),
            'orth_loss': orth_loss.item()
        }
        
        return (bpr_loss, self.lambda_orth * orth_loss), params_to_log
