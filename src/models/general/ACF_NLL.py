import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class AnchorReconstructionLayer(nn.Module):
    """
    ACF 스타일: 앵커의 선형결합으로 임베딩을 재구성하는 레이어
    """
    def __init__(self, num_anchors, embedding_dim):
        super(AnchorReconstructionLayer, self).__init__()
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim
        
        # K개의 앵커 벡터 (공유됨)
        self.anchors = nn.Parameter(torch.empty(num_anchors, embedding_dim))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.anchors)
    
    def forward(self, embedding_tensor):
        """
        입력 임베딩을 앵커의 선형결합으로 재구성
        
        Args:
            embedding_tensor: [..., d] shape의 원본 임베딩
        
        Returns:
            reconstructed: [..., d] shape의 재구성된 임베딩
            coefficients: [..., K] shape의 softmax 계수
        """
        # 1. 각 앵커와의 유사도 계산
        logits = torch.einsum('...d,kd->...k', embedding_tensor, self.anchors)
        
        # 2. Softmax로 계수 계산 (확률 분포 강제)
        coefficients = F.softmax(logits, dim=-1)  # sum = 1
        
        # 3. 앵커의 가중합으로 재구성
        reconstructed = torch.einsum('...k,kd->...d', coefficients, self.anchors)
        
        return reconstructed, coefficients
    
    def exclusiveness_loss(self, item_coefficients):
        """
        아이템이 소수의 앵커만 선택하도록 엔트로피 최소화
        
        Args:
            item_coefficients: (batch_size, K) 아이템 계수들
        """
        # H = -Σ p log p
        entropy = -(item_coefficients * torch.log(item_coefficients + 1e-10)).sum(dim=-1)
        return entropy.mean()
    
    def inclusiveness_loss(self, item_coefficients):
        """
        모든 앵커가 골고루 사용되도록 전역 분포의 엔트로피 최대화
        
        Args:
            item_coefficients: (batch_size, K) 아이템 계수들
        """
        # 전체 아이템에서 각 앵커 사용 비율
        q_k = item_coefficients.mean(dim=0)  # (K,)
        
        # 전역 분포의 엔트로피
        entropy = -(q_k * torch.log(q_k + 1e-10)).sum()
        
        # 최대화하려면 음수 부호 (loss를 minimize)
        return -entropy


class ACF_BPR(BaseModel):
    """
    Anchor-based Collaborative Filtering (ACF) + BPR Loss
    
    핵심 차이점 (vs CSAR):
    1. Softmax 정규화 (크기 정보 소실)
    2. 앵커로 재구성된 임베딩 사용
    3. Entropy 기반 제약 (Exclusiveness + Inclusiveness)
    """
    def __init__(self, config, data_loader):
        super(ACF_BPR, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_anchors = self.config['model']['num_anchors']  # K
        self.lambda_exc = self.config['model'].get('lambda_exclusiveness', 0.01)
        self.lambda_inc = self.config['model'].get('lambda_inclusiveness', 0.01)
        
        # 원본 임베딩 (raw embeddings)
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # 앵커 재구성 레이어
        self.anchor_layer = AnchorReconstructionLayer(self.num_anchors, self.embedding_dim)
        
        self._init_weights()
        self.loss_fn = nn.LogSigmoid()  # BPR loss용
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, users):
        """
        전체 아이템에 대한 점수 계산 (평가용)
        """
        # 원본 임베딩
        user_embs_raw = self.user_embedding(users)
        all_item_embs_raw = self.item_embedding.weight
        
        # 재구성된 임베딩
        user_embs_recon, _ = self.anchor_layer(user_embs_raw)
        item_embs_recon, _ = self.anchor_layer(all_item_embs_raw)
        
        # 재구성된 임베딩 간 내적 (d차원 공간)
        scores = torch.matmul(user_embs_recon, item_embs_recon.T)
        
        return scores
    
    def predict_for_pairs(self, user_ids, item_ids):
        """
        특정 user-item 쌍에 대한 점수 계산
        """
        user_embs_raw = self.user_embedding(user_ids)
        item_embs_raw = self.item_embedding(item_ids)
        
        user_embs_recon, _ = self.anchor_layer(user_embs_raw)
        item_embs_recon, _ = self.anchor_layer(item_embs_raw)
        
        scores = (user_embs_recon * item_embs_recon).sum(dim=-1)
        
        return scores
    
    def get_final_item_embeddings(self):
        """
        최종 아이템 표현 반환 (재구성된 임베딩)
        """
        all_item_embs_raw = self.item_embedding.weight
        item_embs_recon, _ = self.anchor_layer(all_item_embs_raw)
        return item_embs_recon.detach()
    
    def calc_loss(self, batch_data):
        """
        ACF 손실 계산: BPR + Exclusiveness + Inclusiveness
        """
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        # --- 원본 임베딩 ---
        user_embs_raw = self.user_embedding(users)
        pos_item_embs_raw = self.item_embedding(pos_items)
        neg_item_embs_raw = self.item_embedding(neg_items)
        
        # --- 재구성된 임베딩 + 계수 ---
        user_embs_recon, user_coeff = self.anchor_layer(user_embs_raw)
        pos_item_embs_recon, pos_coeff = self.anchor_layer(pos_item_embs_raw)
        neg_item_embs_recon, neg_coeff = self.anchor_layer(neg_item_embs_raw)
        
        # --- 1. BPR Loss ---
        pos_scores = (user_embs_recon * pos_item_embs_recon).sum(dim=-1)
        neg_scores = (user_embs_recon * neg_item_embs_recon).sum(dim=-1)
        
        bpr_loss = -self.loss_fn(pos_scores - neg_scores).mean()
        
        # --- 2. Exclusiveness Loss (아이템만 적용) ---
        # 포지티브와 네거티브 아이템 계수 합쳐서
        item_coeffs = torch.cat([pos_coeff, neg_coeff], dim=0)
        exc_loss = self.anchor_layer.exclusiveness_loss(item_coeffs)
        
        # --- 3. Inclusiveness Loss (전체 배치 기준) ---
        inc_loss = self.anchor_layer.inclusiveness_loss(item_coeffs)
        
        # 로깅용 파라미터
        params_to_log = {
            'bpr_loss': bpr_loss.item(),
            'exclusiveness_loss': exc_loss.item(),
            'inclusiveness_loss': inc_loss.item(),
            'avg_coeff_entropy': -(item_coeffs * torch.log(item_coeffs + 1e-10)).sum(dim=-1).mean().item()
        }
        
        return (bpr_loss,self.lambda_exc * exc_loss,self.lambda_inc * inc_loss), params_to_log
    
    def __str__(self):
        return f"ACF_BPR(num_anchors={self.num_anchors}, embedding_dim={self.embedding_dim})"


class ACF_NLL(BaseModel):
    """
    Anchor-based Collaborative Filtering (ACF) + NLL Loss
    
    원 논문 방식: Full Softmax over all items
    """
    def __init__(self, config, data_loader):
        super(ACF_NLL, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_anchors = self.config['model']['num_anchors']
        self.lambda_exc = self.config['model'].get('lambda_exclusiveness', 0.01)
        self.lambda_inc = self.config['model'].get('lambda_inclusiveness', 0.01)
        
        # 원본 임베딩
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # 앵커 재구성 레이어
        self.anchor_layer = AnchorReconstructionLayer(self.num_anchors, self.embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, users):
        """전체 아이템에 대한 점수"""
        user_embs_raw = self.user_embedding(users)
        all_item_embs_raw = self.item_embedding.weight
        
        user_embs_recon, _ = self.anchor_layer(user_embs_raw)
        item_embs_recon, _ = self.anchor_layer(all_item_embs_raw)
        
        scores = torch.matmul(user_embs_recon, item_embs_recon.T)
        return scores
    
    def predict_for_pairs(self, user_ids, item_ids):
        """특정 쌍 점수"""
        user_embs_raw = self.user_embedding(user_ids)
        item_embs_raw = self.item_embedding(item_ids)
        
        user_embs_recon, _ = self.anchor_layer(user_embs_raw)
        item_embs_recon, _ = self.anchor_layer(item_embs_raw)
        
        scores = (user_embs_recon * item_embs_recon).sum(dim=-1)
        return scores
    
    def get_final_item_embeddings(self):
        all_item_embs_raw = self.item_embedding.weight
        item_embs_recon, _ = self.anchor_layer(all_item_embs_raw)
        return item_embs_recon.detach()
    
    def calc_loss(self, batch_data):
        """
        ACF 원 논문 손실: NLL + Exclusiveness + Inclusiveness
        """
        users = batch_data['user_id']
        items = batch_data['item_id']
        
        # --- 전체 아이템에 대한 점수 계산 (NLL용) ---
        user_embs_raw = self.user_embedding(users)
        all_item_embs_raw = self.item_embedding.weight
        
        user_embs_recon, user_coeff = self.anchor_layer(user_embs_raw)
        all_item_embs_recon, all_item_coeff = self.anchor_layer(all_item_embs_raw)
        
        # --- 1. NLL Loss (Softmax Cross-Entropy) ---
        # 모든 아이템에 대한 점수
        all_scores = torch.matmul(user_embs_recon, all_item_embs_recon.T)  # (batch, n_items)
        
        # Cross-entropy: -log(exp(s_pos) / Σexp(s_all))
        nll_loss = F.cross_entropy(all_scores, items)
        
        # --- 2. Exclusiveness Loss (아이템만) ---
        # 포지티브 아이템의 계수
        pos_coeff = all_item_coeff[items]
        exc_loss = self.anchor_layer.exclusiveness_loss(pos_coeff)
        
        # --- 3. Inclusiveness Loss (전체 아이템 기준) ---
        inc_loss = self.anchor_layer.inclusiveness_loss(all_item_coeff)
        
        
        # 로깅
        params_to_log = {
            'nll_loss': nll_loss.item(),
            'exclusiveness_loss': exc_loss.item(),
            'inclusiveness_loss': inc_loss.item(),
            'avg_anchor_usage': all_item_coeff.mean(dim=0).std().item()  # 앵커 사용 불균형
        }
        
        return (nll_loss,self.lambda_exc * exc_loss,self.lambda_inc * inc_loss), params_to_log
    
    def __str__(self):
        return f"ACF_NLL(num_anchors={self.num_anchors}, embedding_dim={self.embedding_dim})"
