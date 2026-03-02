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
        
        # Loss configuration
        self.loss_type = config.get('train', {}).get('loss_type', 'pairwise')
        self.w_mse = config.get('train', {}).get('w_mse', 10.0)
        
        if self.loss_type == 'w_mse':
            self.loss_fn = nn.MSELoss(reduction='none') # weight 적용을 위해 none으로 설정
            self._log(f"Using All-Item Weighted MSE (w_mse={self.w_mse})")
        else:
            self.loss_fn = BPRLoss()
            self._log("Using pairwise BPRLoss")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def predict_for_pairs(self, users, items):
        user_embeds = self.user_embedding(users)  # [B, D]
        item_embeds = self.item_embedding(items)  # [B, D] or [B, N, D]
        
        return torch.sum(user_embeds * item_embeds, dim=-1)

    def calc_loss(self, batch_data):
        if self.loss_type == 'pointwise':
            # All-Item Weighted MSE (WMF 스타일)
            users = batch_data['user_id'] # [B]
            pos_items = batch_data['item_id'] # [B]
            
            # [Optimization] 임베딩 직접 조회 (L2 위해)
            u_emb = self.user_embedding(users)
            
            # 1. 모든 아이템에 대한 점수 계산
            all_scores = torch.matmul(u_emb, self.item_embedding.weight.transpose(0, 1))
            
            # 2. 타겟 생성 (0으로 초기화 후 긍정 아이템만 1로 설정)
            targets = torch.zeros_like(all_scores)
            targets.scatter_(1, pos_items.unsqueeze(1), 1.0)
            
            # 3. 가중치 생성 (긍정 아이템은 w_mse, 나머지는 1.0)
            weights = targets * (self.w_mse - 1.0) + 1.0
            
            # 4. Weighted MSE 로스 계산
            loss = (self.loss_fn(all_scores, targets) * weights).mean()
            
            # [추가] L2 규제 (사용된 임베딩만)
            l2_loss = self.get_l2_reg_loss(u_emb, self.item_embedding(pos_items))
        else:
            # Pairwise (BPR)
            users = batch_data['user_id']
            pos_items = batch_data['pos_item_id']
            neg_items = batch_data['neg_item_id']

            u_emb = self.user_embedding(users)
            p_emb = self.item_embedding(pos_items)
            n_emb = self.item_embedding(neg_items)

            pos_scores = torch.sum(u_emb * p_emb, dim=-1)
            neg_scores = torch.sum(u_emb * n_emb, dim=-1)
        
            loss = self.loss_fn(pos_scores, neg_scores)
            
            # [추가] L2 규제
            l2_loss = self.get_l2_reg_loss(u_emb, p_emb, n_emb)
        
        return (loss, l2_loss), {'loss_main': loss.item(), 'loss_l2': l2_loss.item()}

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
