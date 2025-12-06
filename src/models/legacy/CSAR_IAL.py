import torch
import torch.nn as nn
import torch.nn.functional as F
from .CSAR_R_BPR import CSAR_R_BPR

class CSAR_IAL(CSAR_R_BPR):
    """
    CSAR_IAL (Interest Alignment Loss)
    
    CSAR_R_BPR를 상속받아, 유저와 포지티브 아이템 간의 
    관심사 분포(Interest Distribution)를 일치시키는 Alignment Loss를 추가한 모델.
    
    Loss = BPR Loss + lambda * KL_Divergence(User_Interest || Item_Interest)
    """
    def __init__(self, config, data_loader):
        super(CSAR_IAL, self).__init__(config, data_loader)
        self.alignment_weight = self.config['model'].get('alignment_weight', 0.1)
        
    def calc_loss(self, batch_data):
        # DataLoader가 [B, 1] 형태로 반환하므로 차원 축소
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        # --- 1. 임베딩 및 관심사 추출 ---
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)

        # 관심사 벡터 (Softplus 통과된 비음수 값)
        user_interests = self.attention_layer(user_embs)       # [B, K]
        pos_item_interests = self.attention_layer(pos_item_embs) # [B, K]
        neg_item_interests = self.attention_layer(neg_item_embs) # [B, K]

        # --- 2. BPR Loss 계산 (기존 로직) ---
        # Positive Score
        pos_interest_scores = (user_interests * pos_item_interests).sum(dim=-1)
        pos_res_scores = (user_embs * pos_item_embs).sum(dim=-1)
        pos_scores = pos_interest_scores + pos_res_scores

        # Negative Score
        neg_interest_scores = (user_interests * neg_item_interests).sum(dim=-1)
        neg_res_scores = (user_embs * neg_item_embs).sum(dim=-1)
        neg_scores = neg_interest_scores + neg_res_scores
        
        bpr_loss = self.loss_fn(pos_scores, neg_scores)

        # --- 3. Interest Alignment Loss (KL Divergence) ---
        # 관심사 벡터를 확률 분포로 변환 (Normalize)
        # epsilon 추가하여 log(0) 방지
        eps = 1e-10
        p_user = user_interests / (user_interests.sum(dim=-1, keepdim=True) + eps)
        q_item = pos_item_interests / (pos_item_interests.sum(dim=-1, keepdim=True) + eps)
        
        # KL(P || Q) = sum(P * log(P/Q))
        # P: User Interest (Target Distribution - 유저가 진정 원하는 것)
        # Q: Item Interest (Approximation - 아이템이 제공하는 것)
        # 유저의 관심사 분포를 아이템이 얼마나 잘 커버하는지 측정
        kl_loss = F.kl_div(q_item.log(), p_user, reduction='batchmean', log_target=False)
        
        # --- 4. Orthogonal Loss ---
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
    
        
        # 로깅
        params_to_log = {
            'bpr_loss': bpr_loss.item(),
            'align_loss': kl_loss.item(),
            'orth_loss': orth_loss.item()
        }
        if isinstance(self.attention_layer.scale, nn.Parameter):
            params_to_log['scale'] = self.attention_layer.scale.item()

        return (bpr_loss,self.alignment_weight * kl_loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_IAL(num_interests={self.num_interests}, align_w={self.alignment_weight})"
