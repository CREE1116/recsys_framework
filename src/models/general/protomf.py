import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from src.loss import BPRLoss

class ProtoMF(BaseModel):
    """
    ProtoMF: Prototype-based Matrix Factorization (RecSys'22)
    공식 구현 참고: https://github.com/hcai-mms/ProtoMF
    """
    def __init__(self, config, data_loader):
        super(ProtoMF, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_prototypes = int(self.config['model']['num_prototypes'])
        self.mode = self.config['model'].get('mode', 'UI')  # 'U', 'I', or 'UI'
        
        # Inclusion Regularization weights (원본 구현)
        self.sim_proto_weight = float(self.config['model'].get('sim_proto_weight', 1.0))
        self.sim_batch_weight = float(self.config['model'].get('sim_batch_weight', 1.0))
        
        # Cosine type: 'shifted' (0~2), 'shifted_and_div' (0~1), 'standard' (-1~1)
        self.cosine_type = self.config['model'].get('cosine_type', 'shifted_and_div')
        
        # Raw embeddings
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # Prototypes (공식 구현과 동일)
        self.user_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim))
        self.item_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim))
        
        # Transformation matrices (논문의 W_u, W_t)
        self.W_u = nn.Linear(self.embedding_dim, self.num_prototypes, bias=False)
        self.W_t = nn.Linear(self.embedding_dim, self.num_prototypes, bias=False)
        
        # Cosine Similarity Function (공식 구현 방식)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.loss_fn = BPRLoss()
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.W_u.weight)
        nn.init.xavier_normal_(self.W_t.weight)
        # Prototypes는 randn으로 이미 초기화됨 (공식 구현과 동일)

    def prototype_similarity(self, embeddings, prototypes):
        """
        공식 구현의 cosine_sim_func 방식
        embeddings: [B, D] or [M, D]
        prototypes: [K, D]
        return: [B, K] or [M, K]
        """
        # Normalize vectors (L2 norm)
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # Matrix Multiplication: [B, D] @ [D, K] -> [B, K]
        cos_sim = torch.matmul(emb_norm, proto_norm.t())
        
        if self.cosine_type == 'standard':
            return cos_sim  # [-1, 1]
        elif self.cosine_type == 'shifted':
            return 1 + cos_sim  # [0, 2]
        else:  # 'shifted_and_div' (가장 안정적)
            return (1 + cos_sim) / 2  # [0, 1]

    def _inclusion_regularizer(self, sim_mtx):
        """
        Inclusion criteria (max-sim based), as described in ProtoMF.
        sim_mtx: [B, K] similarities between batch embeddings and prototypes.

        - batch_inclusion: each sample should match at least one prototype
          => maximize max_k sim[b, k]
        - proto_inclusion: each prototype should match at least one sample in the batch
          => maximize max_b sim[b, k]

        We return losses to MINIMIZE, hence negatives.
        """
        # [B]
        max_over_proto = sim_mtx.max(dim=1).values
        # [K]
        max_over_batch = sim_mtx.max(dim=0).values

        batch_inclusion_loss = -max_over_proto.mean()
        proto_inclusion_loss = -max_over_batch.mean()
        return batch_inclusion_loss, proto_inclusion_loss

    def forward(self, users):
        u_emb = self.user_embedding(users)  # [B, D]
        i_emb_all = self.item_embedding.weight  # [M, D]
        
        scores = torch.zeros(users.size(0), self.data_loader.n_items, device=users.device)
        
        # U-ProtoMF: User를 Prototype과 비교
        if self.mode in ['U', 'UI']:
            u_star = self.prototype_similarity(u_emb, self.user_prototypes)  # [B, K]
            t_tilde = self.W_t(i_emb_all)  # [M, K]
            u_score = torch.matmul(u_star, t_tilde.T)  # [B, M]
            scores = scores + u_score
        
        # I-ProtoMF: Item을 Prototype과 비교
        if self.mode in ['I', 'UI']:
            t_star = self.prototype_similarity(i_emb_all, self.item_prototypes)  # [M, K]
            u_tilde = self.W_u(u_emb)  # [B, K]
            i_score = torch.matmul(u_tilde, t_star.T)  # [B, M]
            scores = scores + i_score
        
        return scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        if pos_items.ndim > 1:
            pos_items = pos_items.squeeze()
            
        pos_scores = self.predict_for_pairs(users, pos_items)
        
        if neg_items.ndim == 1:
            neg_scores = self.predict_for_pairs(users, neg_items)
            pos_scores = pos_scores.view(-1, 1)
            neg_scores = neg_scores.view(-1, 1)
        else:
            B, K = neg_items.shape
            users_for_neg = users.repeat_interleave(K)
            neg_items_flat = neg_items.view(-1)
            neg_scores_flat = self.predict_for_pairs(users_for_neg, neg_items_flat)
            neg_scores = neg_scores_flat.view(B, K)
            pos_scores = pos_scores.view(B, 1)

        # BPR Loss (공식 구현과 동일)
        bpr_loss = self.loss_fn(pos_scores,neg_scores)

        # Inclusion Regularization (ProtoMF-style: max similarity criteria)
        reg_loss = torch.zeros((), device=users.device)

        # User side
        if self.mode in ['U']:
            u_emb = self.user_embedding(users)  # [B, D]
            u_sim = self.prototype_similarity(u_emb, self.user_prototypes)  # [B, K]
            batch_inc_u, proto_inc_u = self._inclusion_regularizer(u_sim)
            reg_loss = reg_loss + self.sim_batch_weight * batch_inc_u + self.sim_proto_weight * proto_inc_u
            return (bpr_loss, reg_loss), {}

        # Item side
        if self.mode in ['I']:
            pos_emb = self.item_embedding(pos_items)  # [B, D]
            t_sim = self.prototype_similarity(pos_emb, self.item_prototypes)  # [B, K]
            batch_inc_t, proto_inc_t = self._inclusion_regularizer(t_sim)
            reg_loss = reg_loss + self.sim_batch_weight * batch_inc_t + self.sim_proto_weight * proto_inc_t
            return (bpr_loss, reg_loss), {}

        # both side
        if self.mode in ['UI']:
            u_emb = self.user_embedding(users)  # [B, D]
            u_sim = self.prototype_similarity(u_emb, self.user_prototypes)  # [B, K]
            batch_inc_u, proto_inc_u = self._inclusion_regularizer(u_sim)
            user_reg_loss = self.sim_batch_weight * batch_inc_u + self.sim_proto_weight * proto_inc_u

            pos_emb = self.item_embedding(pos_items)  # [B, D]
            t_sim = self.prototype_similarity(pos_emb, self.item_prototypes)  # [B, K]
            batch_inc_t, proto_inc_t = self._inclusion_regularizer(t_sim)
            item_reg_loss = self.sim_batch_weight * batch_inc_t + self.sim_proto_weight * proto_inc_t
            return (bpr_loss, user_reg_loss, item_reg_loss), {}
    


    def predict_for_pairs(self, user_ids, item_ids):
        """User-Item 쌍의 점수 계산"""
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        
        scores = torch.zeros(len(user_ids), device=user_ids.device)
        
        # U-ProtoMF
        if self.mode in ['U', 'UI']:
            u_star = self.prototype_similarity(u_emb, self.user_prototypes)  # [B, K]
            t_tilde = self.W_t(i_emb)  # [B, K]
            scores = scores + (u_star * t_tilde).sum(dim=1)
        
        # I-ProtoMF
        if self.mode in ['I', 'UI']:
            t_star = self.prototype_similarity(i_emb, self.item_prototypes)  # [B, K]
            u_tilde = self.W_u(u_emb)  # [B, K]
            scores = scores + (u_tilde * t_star).sum(dim=1)
            
        return scores

    def get_final_item_embeddings(self):
        """ILD 등 메트릭용 아이템 임베딩 반환"""
        return self.item_embedding.weight.detach()
    
    def __str__(self):
        return f"ProtoMF(K={self.num_prototypes}, mode={self.mode}, cosine={self.cosine_type})"