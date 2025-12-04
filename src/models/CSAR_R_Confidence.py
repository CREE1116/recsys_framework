import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_R_Confidence(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_R_Confidence, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1)
        self.scale = self.config['model'].get('scale', True)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale)
        
        # [Gating Params]
        self.entropy_threshold = nn.Parameter(torch.tensor(0.5)) 
        self.gating_scale = 10.0

        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _calc_confidence(self, interests):
        # [안전장치 1] 입력값 안정화 (Logits가 너무 크면 Softmax 터짐)
        # 최대값을 빼주는 테크닉(Log-Sum-Exp trick의 변형)을 써도 되지만,
        # 간단하게는 값을 10 이하로 눌러주는 Tanh나 Clamp를 쓸 수 있음.
        # 여기서는 값이 너무 커지지 않게 스케일링만 살짝 함.
        interests = interests / (interests.norm(dim=1, keepdim=True) + 1e-9) * 10.0

        # 1. 확률 분포
        probs = F.softmax(interests, dim=1) 
        
        # 2. 엔트로피 계산 (Safe Log)
        probs = torch.clamp(probs, min=1e-9, max=1.0)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        
        # 3. 정규화된 엔트로피 (0 ~ 1)
        max_entropy = torch.log(torch.tensor(float(self.num_interests), device=interests.device))
        norm_entropy = entropy / max_entropy
        
        # [안전장치 2] Threshold 파라미터가 미쳐 날뛰지 않게 제어
        # 학습 중에는 clamp_를 쓰거나, 사용할 때 sigmoid를 씌워 0~1로 매핑해서 사용
        # 여기서는 sigmoid를 씌워 '유효 범위'를 강제함
        effective_threshold = torch.sigmoid(self.entropy_threshold) 

        # 4. 신뢰도(Alpha) 계산
        # alpha = Sigmoid( Scale * (Threshold - Entropy) )
        # Scale이 너무 크면(10.0) 기울기가 가팔라서 0 아니면 1로 튈 수 있음. 
        # 5.0 정도로 줄이는 것도 방법.
        logit = self.gating_scale * (effective_threshold - norm_entropy)
        alpha = torch.sigmoid(logit)
        
        return alpha
    
    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def forward(self, users):
        # ... (기존과 동일) ...
        u_emb = self.user_embedding(users)
        all_i_emb = self.item_embedding.weight

        mf_score = torch.matmul(u_emb, all_i_emb.t())

        u_int = self.attention_layer(u_emb)
        i_int = self.attention_layer(all_i_emb)
        csar_score = torch.einsum('bk,nk->bn', u_int, i_int)
        
        alpha = self._calc_confidence(u_int)
        
        return mf_score + (alpha * csar_score)

    def predict_for_pairs(self, user_ids, item_ids):
        """
        [수정] 학습/평가 로직 일원화
        여기서 계산된 alpha가 학습에도 그대로 반영되어야 함
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # 1. MF Score
        mf_score = (u_emb * i_emb).sum(dim=-1)

        # 2. CSAR Score
        u_int = self.attention_layer(u_emb)
        i_int = self.attention_layer(i_emb)
        csar_score = (u_int * i_int).sum(dim=-1)

        # 3. Confidence Gating
        # [B, K] -> [B]
        alpha = self._calc_confidence(u_int).squeeze(-1)

        return mf_score + (alpha * csar_score)

    def calc_loss(self, batch_data):
        """
        [수정] 수동 계산 대신 predict_for_pairs 호출로 일관성 확보 및 NaN 방지
        """
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        # 1. 점수 계산 (일관된 로직 사용)
        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
        
        # [NaN 방지 Tip] 점수가 너무 커지면 BPR exp 연산에서 터질 수 있음.
        # 필요하다면 점수를 살짝 스케일링하거나(예: * 0.1), BPR Loss 구현체 내부에서 LogSigmoid 사용 확인.
        
        # 2. BPR Loss
        loss = self.loss_fn(pos_scores, neg_scores)

        # 3. Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l2")
        
        params_to_log = {'scale':self.attention_layer.scale.item(),'alpha_threshold':torch.sigmoid(self.entropy_threshold).item()}
        
        return (loss, self.lamda_orth * orth_loss), params_to_log