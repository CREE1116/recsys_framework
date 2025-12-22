import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class ProtoMF_Sampled(BaseModel):
    """
    ProtoMF with Sampled Softmax Loss (InfoNCE)
    기존 BPR 대신 NormalizedSampledSoftmaxLoss 사용
    """
    def __init__(self, config, data_loader):
        super(ProtoMF_Sampled, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_prototypes = int(self.config['model']['num_prototypes'])
        
        # Inclusion Regularization weights
        self.sim_proto_weight = float(self.config['model'].get('sim_proto_weight', 1.0))
        self.sim_batch_weight = float(self.config['model'].get('sim_batch_weight', 1.0))
        
        self.mode = self.config['model'].get('mode', 'UI')  # 'U', 'I', or 'UI'
        
        # Cosine type: 'shifted' (0~2), 'shifted_and_div' (0~1), 'standard' (-1~1)
        self.cosine_type = self.config['model'].get('cosine_type', 'shifted_and_div')
        
        # Sampled Softmax parameters
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.is_explicit = self.num_negatives > 0
        
        # Raw embeddings
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # Prototypes
        self.user_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim))
        self.item_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim))
        
        # Transformation matrices
        self.W_u = nn.Linear(self.embedding_dim, self.num_prototypes, bias=False)
        self.W_t = nn.Linear(self.embedding_dim, self.num_prototypes, bias=False)
        
        # Cosine Similarity Function
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        
        self._init_weights()
        
        # NormalizedSampledSoftmaxLoss (Sampled Softmax)
        temperature = config['model'].get('temperature', 1.0)
        from src.loss import NormalizedSampledSoftmaxLoss
        self.loss_fn = NormalizedSampledSoftmaxLoss(
            self.data_loader.n_items, 
            temperature=temperature
        )

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.W_u.weight)
        nn.init.xavier_normal_(self.W_t.weight)

    def prototype_similarity(self, embeddings, prototypes):
        """공식 구현의 cosine_sim_func 방식"""
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_sim = torch.matmul(emb_norm, proto_norm.t())
        
        if self.cosine_type == 'standard':
            return cos_sim
        elif self.cosine_type == 'shifted':
            return 1 + cos_sim
        else:  # 'shifted_and_div'
            return (1 + cos_sim) / 2

    def _inclusion_regularizer(self, sim_mtx):
        """
        Inclusion criteria (max-sim based), as described in ProtoMF.
        sim_mtx: [B, K] similarities between batch embeddings and prototypes.
        """
        # [B]
        max_over_proto = sim_mtx.max(dim=1).values
        # [K]
        max_over_batch = sim_mtx.max(dim=0).values

        batch_inclusion_loss = -max_over_proto.mean()
        proto_inclusion_loss = -max_over_batch.mean()
        return batch_inclusion_loss, proto_inclusion_loss

    def _get_user_representation(self, u_emb):
        """User 표현 계산 (U-ProtoMF + I-ProtoMF 경로)"""
        result = torch.zeros(u_emb.size(0), self.num_prototypes, device=u_emb.device)
        
        if self.mode in ['U', 'UI']:
            u_star = self.prototype_similarity(u_emb, self.user_prototypes)
            result = result + u_star
        
        if self.mode in ['I', 'UI']:
            u_tilde = self.W_u(u_emb)
            result = result + u_tilde
            
        return result

    def _get_item_representation(self, i_emb):
        """Item 표현 계산 (U-ProtoMF + I-ProtoMF 경로)"""
        result = torch.zeros(i_emb.size(0), self.num_prototypes, device=i_emb.device)
        
        if self.mode in ['U', 'UI']:
            t_tilde = self.W_t(i_emb)
            result = result + t_tilde
        
        if self.mode in ['I', 'UI']:
            t_star = self.prototype_similarity(i_emb, self.item_prototypes)
            result = result + t_star
            
        return result

    def forward(self, users):
        u_emb = self.user_embedding(users)
        i_emb_all = self.item_embedding.weight
        
        user_repr = self._get_user_representation(u_emb)
        item_repr = self._get_item_representation(i_emb_all)
        
        scores = torch.matmul(user_repr, item_repr.T)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        
        user_repr = self._get_user_representation(u_emb)
        item_repr = self._get_item_representation(i_emb)
        
        scores = (user_repr * item_repr).sum(dim=1)
        return scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        
        user_repr = self._get_user_representation(self.user_embedding(users))
        pos_item_repr = self._get_item_representation(self.item_embedding(pos_items))
        
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            
            B, N = neg_items.size()
            flat_neg_ids = neg_items.view(-1)
            neg_item_repr = self._get_item_representation(self.item_embedding(flat_neg_ids)).view(B, N, -1)
            
            # Pos Scores [B, 1]
            pos_scores = (user_repr * pos_item_repr).sum(dim=-1, keepdim=True)
            
            # Neg Scores [B, N]
            neg_scores = (user_repr.unsqueeze(1) * neg_item_repr).sum(dim=-1)
            
            # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
        else:
            # In-Batch Negatives
            scores = torch.matmul(user_repr, pos_item_repr.t())
        
        loss = self.loss_fn(scores, is_explicit=self.is_explicit)
        
        # Inclusion Regularization (ProtoMF-style: max similarity criteria)
        reg_loss = torch.zeros((), device=users.device)

        # User side
        if self.mode in ['U', 'UI']:
            u_emb = self.user_embedding(users)  # [B, D]
            u_sim = self.prototype_similarity(u_emb, self.user_prototypes)  # [B, K]
            batch_inc_u, proto_inc_u = self._inclusion_regularizer(u_sim)
            reg_loss = reg_loss + self.sim_batch_weight * batch_inc_u + self.sim_proto_weight * proto_inc_u

        # Item side
        if self.mode in ['I', 'UI']:
            pos_emb = self.item_embedding(pos_items)  # [B, D]
            t_sim = self.prototype_similarity(pos_emb, self.item_prototypes)  # [B, K]
            batch_inc_t, proto_inc_t = self._inclusion_regularizer(t_sim)
            reg_loss = reg_loss + self.sim_batch_weight * batch_inc_t + self.sim_proto_weight * proto_inc_t
        
        return (loss, reg_loss), {}

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()
    
    def __str__(self):
        return f"ProtoMF_Sampled(K={self.num_prototypes}, mode={self.mode})"
