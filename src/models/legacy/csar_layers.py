import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoSupportAttentionLayer(nn.Module):
    """
    d-차원의 임베딩을 K-차원의 비음수 관심사 가중치 벡터로 변환하는 레이어.
    """
    def __init__(self, num_interests, embedding_dim, scale=False, normalize=False, init_method = "xavier"):
        super(CoSupportAttentionLayer, self).__init__() 
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.init_method = init_method
        # K-Anchor (관심사 키)
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        init_scale = num_interests ** -0.5
        if scale:
            self.scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer("scale", torch.tensor(init_scale))
        
        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        gain = 1.45
        """Xavier uniform 초기화를 사용하여 관심사 키를 초기화합니다."""
        if self.init_method == "xavier":
            nn.init.xavier_uniform_(self.interest_keys, gain=gain)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.interest_keys, gain=gain)
        
    def forward(self, embedding_tensor):
        """
        입력 텐서를 K-차원 관심사 가중치로 변환합니다.
        
        Args:
            embedding_tensor (torch.Tensor): [..., d] shape의 임베딩 텐서.
        
        Returns:
            torch.Tensor: [..., K] shape의 비음수 관심사 가중치 텐서.
        """
        # scale이 Parameter인 경우와 Tensor인 경우 모두 처리
        attention_logits =F.softplus(torch.einsum('...d,kd->...k', embedding_tensor, self.interest_keys) * self.scale)
        if self.normalize:
            return F.normalize(attention_logits, p=2, dim=-1)
        return attention_logits 
    
    @staticmethod
    def l1_orthogonal_loss(keys):
        """L1 norm을 사용하여 관심사 키의 직교성을 강제합니다."""
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        
        # 비대각선 요소의 개수
        num_off_diagonal_elements = K * K - K
        
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보
        loss = torch.abs(cosine_similarity - identity_matrix).sum()
        return loss / num_off_diagonal_elements 
    
    @staticmethod
    def l2_orthogonal_loss(keys):
        """L2 norm을 사용하여 관심사 키의 직교성을 강제합니다."""
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        off_diag = cosine_similarity - identity_matrix
        
        # 1. 제곱의 합 (L2 Loss)
        l2_sum_loss = (off_diag ** 2).sum()
        
        # 2. 비대각선 요소의 개수로 나누어 정규화
        num_off_diagonal_elements = K * K - K
        normalized_loss = l2_sum_loss / num_off_diagonal_elements 
        return normalized_loss 

    def get_orth_loss(self, loss_type="l2"):
        """이 레이어가 소유한 관심사 키에 대한 직교 손실을 반환합니다."""
        interest_keys = self.interest_keys
        if loss_type == "l1":
            return self.l1_orthogonal_loss(interest_keys)
        else:
            return self.l2_orthogonal_loss(interest_keys)



class AdaptiveContrastiveLoss(nn.Module):
    """
    [Black Box Module]
    Input: 원본 관심사 벡터 (interests), 관심사 키 (keys)
    Output: Contrastive Loss (Scalar)
    
    기능:
    1. Adaptive Tau 계산 (Entropy & Orthogonality 기반, No-Grad 안전장치 포함)
    2. Data Augmentation (Noise vs Noise)
    3. Unidirectional InfoNCE 계산
    """
    def __init__(self, noise_sigma=0.1, base_tau=0.1, alpha=0.5):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.noise_sigma = noise_sigma
        self.base_tau = base_tau
        self.alpha = alpha

    def _get_adaptive_tau(self, interests, keys):
        """내부용: 현재 상태에 따른 Temperature 계산"""
        B, K = interests.size()
        with torch.no_grad(): # 안전장치: Tau 조작 방지
            # 1. Uncertainty (Entropy)
            p = F.softmax(interests, dim=1)
            log_p = F.log_softmax(interests, dim=1)
            entropy = -(p * log_p).sum(dim=1)
            uncertainty = entropy / math.log(K + 1e-8) # [B]

            # 2. Key Quality (Orthogonality)
            W_norm = F.normalize(keys, dim=1)
            gram = W_norm @ W_norm.T
            eye = torch.eye(K, device=keys.device)
            off_diag_mean = (gram - eye).abs().sum() / (K * (K - 1) + 1e-8)
            quality = 1.0 - off_diag_mean

            # 3. Calc Tau
            net_difficulty = uncertainty - quality
            tau = self.base_tau * torch.exp(self.alpha * net_difficulty)
            return torch.clamp(tau, min=0.05, max=0.5).unsqueeze(1)

    def forward(self, interests, keys):
        """
        사용법: loss = self.cl_layer(user_interests, self.attention.keys)
        """
        # 1. Tau 계산
        tau = self._get_adaptive_tau(interests, keys)

        # 2. Augmentation (Noise vs Noise)
        # 원본이 Softplus를 통과한 양수이므로, 노이즈 후에도 양수 성질 유지(Softplus) 추천
        noise1 = torch.randn_like(interests) * self.noise_sigma
        noise2 = torch.randn_like(interests) * self.noise_sigma
        
        view1 = F.softplus(interests + noise1)
        view2 = F.softplus(interests + noise2)

        # 3. InfoNCE (Unidirectional for efficiency)
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        
        logits = (z1 @ z2.T) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        
        return F.cross_entropy(logits, labels)

class DualViewCoSupportAttentionLayer(nn.Module):
    """
    긍정,부정 뷰를 제공하는 CSA레이어
    Positive View: 좋아하는 관심사
    Negative View: 싫어하는 관심사
    Score = Pos - Neg
    """
    def __init__(self, num_interests, embedding_dim, scale=False, normalize=False):
        super(DualViewCoSupportAttentionLayer, self).__init__() 
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        # K-Anchor (관심사 키) - Pos/Neg 각각 K개
        self.pos_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        self.neg_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        init_val = float(num_interests)**-0.5
        if scale:
            self.pos_scale = nn.Parameter(torch.tensor(init_val))
            self.neg_scale = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer("pos_scale", torch.tensor(init_val))
            self.register_buffer("neg_scale", torch.tensor(init_val))
            
        # 가중치 초기화
        self._init_weights()


    def _init_weights(self):
        """Xavier uniform 초기화를 사용하여 관심사 키를 초기화합니다."""
        nn.init.xavier_uniform_(self.pos_keys)
        nn.init.xavier_uniform_(self.neg_keys)  
        
    def forward(self, embedding_tensor):
        """
        입력 텐서를 K-차원 관심사 가중치로 변환합니다.
        
        Args:
            embedding_tensor (torch.Tensor): [..., d] shape의 임베딩 텐서.
        
        Returns:
            torch.Tensor: [..., K] shape의 비음수 관심사 가중치 텐서.
        """
        # scale이 Parameter인 경우와 Tensor인 경우 모두 처리
        pos_logits =F.softplus(torch.einsum('...d,kd->...k', embedding_tensor, self.pos_keys) * self.pos_scale)
        neg_logits =F.softplus(torch.einsum('...d,kd->...k', embedding_tensor, self.neg_keys) * self.neg_scale)
        if self.normalize:
            return F.normalize(pos_logits, p=2, dim=-1), F.normalize(neg_logits, p=2, dim=-1)
        return pos_logits, neg_logits   

    def get_orth_loss(self, loss_type="l2"):
        """이 레이어가 소유한 관심사 키에 대한 직교 손실을 반환합니다."""
        # 1. 두 키를 합칩니다. (Shape: [2*num_interests, embedding_dim])
        all_keys = torch.cat([self.pos_keys, self.neg_keys], dim=0)
        
        # 2. 이 거대한 행렬에 대해 직교 로스를 한 방에 계산합니다.
        # 모든 키가 서로서로 달라지도록 강제합니다.
        
        # 정규화
        keys_norm = F.normalize(all_keys, p=2, dim=-1)
        
        # 내적 (Gram Matrix: [2K, 2K])
        gram_matrix = torch.matmul(keys_norm, keys_norm.t())
        
        # 항등 행렬 (Identity Matrix)
        identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        if loss_type == "l1":
        # 차이의 절대값 합 (L1 Loss 추천 - 희소성 유도에 좋음)
            loss = torch.abs(gram_matrix - identity).sum()
        elif loss_type == "l2":
        # 차이의 제곱합 (L2 Loss 추천 - 수렴 속도 유도에 좋음)
            loss = torch.pow(gram_matrix - identity, 2).sum()
        
        # 요소 개수로 나누어 스케일 맞춤 (선택)
        num_elements = gram_matrix.numel() - gram_matrix.size(0) # 대각선 제외 개수
        return loss / num_elements


class CSAR_CovDino(nn.Module):
    """
    CSAR with Covariance-based G and DINO-style PP alignment (EMA 방식)
    
    핵심 아이디어:
    1. 배치 내 멤버십의 코사인 유사도를 EMA로 업데이트하여 G로 사용
    2. PP (프로토타입 간 유사도)가 이 G를 따라가도록 DINO 스타일 loss
    
    Args:
        num_interests: 관심사 수 (K)
        embedding_dim: 임베딩 차원 (d)
        ema_momentum: EMA 업데이트 모멘텀 (기본: 0.9)
    """
    def __init__(self, num_interests, embedding_dim, ema_momentum=0.9):
        super(CSAR_CovDino, self).__init__()
        self.K = num_interests
        self.d = embedding_dim
        self.ema_momentum = ema_momentum
        
        # Student: 학습되는 프로토타입
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        # Teacher: 멤버십 코사인 유사도의 EMA
        self.register_buffer('ema_G', torch.eye(num_interests))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)

    def get_membership(self, embs):
        """코사인 유사도 멤버십: [-1, 1]"""
        embs_norm = F.normalize(embs, p=2, dim=-1)
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return embs_norm @ keys_norm.t()  # [-1, 1]
    
    @torch.no_grad()
    def update_ema_G(self, memberships):
        """
        배치 멤버십의 코사인 유사도 평균으로 EMA 업데이트
        
        Args:
            memberships: [B, K] (user + item 합친 멤버십)
        """
        # 각 관심사(열)를 L2 정규화 → 관심사 간 코사인 유사도
        a_norm = F.normalize(memberships, p=2, dim=0)  # [B, K], 각 열 norm=1
        
        # 정규화된 관심사 벡터들의 외적 = 코사인 유사도
        sim = a_norm.t() @ a_norm  # [K, K], 대각=1
        
        # EMA 업데이트
        self.ema_G = self.ema_momentum * self.ema_G + (1 - self.ema_momentum) * sim
    
    def get_dino_loss(self):
        """
        PP가 EMA_G를 따라가도록 하는 loss
        
        Returns:
            MSE loss between PP and G
        """
        # PP: Keys의 코사인 유사도
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        PP = keys_norm @ keys_norm.t()  # [K, K], 대각=1
        
        G = self.ema_G.detach()  # [K, K], 대각≈1
        
        return F.mse_loss(PP, G)
    
    def get_gram_matrix(self):
        """Score 계산에는 EMA_G 사용"""
        return self.ema_G
    
    def get_pp_matrix(self):
        """프로토타입 간 코사인 유사도 (대각=1)"""
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return keys_norm @ keys_norm.t()
        
    def forward(self, embs):
        return self.get_membership(embs)


class CSAR_CovBatch(nn.Module):
    """
    CSAR with Batch-Average G (EMA 없이, BatchNorm 스타일)
    
    핵심 아이디어:
    - 학습 시: 현재 배치의 멤버십 코사인 유사도를 G로 직접 사용
    - 추론 시: 학습 중 누적된 running average 사용 (BatchNorm처럼)
    
    Args:
        num_interests: 관심사 수 (K)
        embedding_dim: 임베딩 차원 (d)
        momentum: running average 업데이트 모멘텀 (기본: 0.1, BatchNorm 기본값)
    """
    def __init__(self, num_interests, embedding_dim, momentum=0.1):
        super(CSAR_CovBatch, self).__init__()
        self.K = num_interests
        self.d = embedding_dim
        self.momentum = momentum
        
        # 프로토타입
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        # Running average (추론용)
        self.register_buffer('running_G', torch.eye(num_interests))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # 현재 배치의 G (학습용)
        self.register_buffer('batch_G', torch.eye(num_interests))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)

    def get_membership(self, embs):
        """코사인 유사도 멤버십: [-1, 1]"""
        embs_norm = F.normalize(embs, p=2, dim=-1)
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return embs_norm @ keys_norm.t()
    
    def update_batch_G(self, memberships):
        """
        배치 멤버십으로 G 계산 (학습 시 호출)
        
        Args:
            memberships: [B, K] (user + item 합친 멤버십)
        """
        # 각 관심사(열)를 L2 정규화 → 관심사 간 코사인 유사도
        a_norm = F.normalize(memberships, p=2, dim=0)  # [B, K], 각 열 norm=1
        
        # 정규화된 관심사 벡터들의 외적 = 코사인 유사도
        batch_sim = a_norm.t() @ a_norm  # [K, K], 대각=1
        
        # 현재 배치 G 저장 (학습 시 사용)
        self.batch_G = batch_sim
        
        # Running average 업데이트 (추론용)
        with torch.no_grad():
            self.num_batches_tracked += 1
            if self.num_batches_tracked == 1:
                self.running_G = batch_sim.detach()
            else:
                self.running_G = (1 - self.momentum) * self.running_G + self.momentum * batch_sim.detach()
    
    def get_gram_matrix(self):
        """학습 시 배치 G, 추론 시 running G 반환"""
        if self.training:
            return self.batch_G
        else:
            return self.running_G
    
    def get_pp_matrix(self):
        """프로토타입 간 코사인 유사도 (대각=1)"""
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return keys_norm @ keys_norm.t()
    
    def get_dino_loss(self):
        """PP가 batch_G를 따라가도록 하는 loss"""
        PP = self.get_pp_matrix()
        G = self.batch_G.detach()
        return F.mse_loss(PP, G)
        
    def forward(self, embs):
        return self.get_membership(embs)
    def forward(self, embs):
        return self.get_membership(embs)


class CSAR_basic(nn.Module):
    """
    CSAR Basic Layer
    Softplus 기반의 간단한 Attention 메커니즘 사용
    """
    def __init__(self, num_interests, embedding_dim):
        super(CSAR_basic, self).__init__()
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        
        # Interest Keys
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        # Scale parameter
        self.scale = nn.Parameter(torch.tensor(num_interests ** -0.5))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)
        
    def get_membership(self, embs):
        """
        Args:
            embs: [..., d]
        Returns:
            attention_logits: [..., K] (Non-negative)
        """
        # Softplus activation for non-negative membership
        logits = torch.einsum('...d,kd->...k', embs, self.interest_keys) * self.scale
        return F.softplus(logits)
        
    def forward(self, embedding_tensor):
        return self.get_membership(embedding_tensor)
    
    def get_gram_matrix(self):
        """
        Prototype Correlation Matrix (G = P @ P.T)
        """
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return keys_norm @ keys_norm.t()  # [K, K]
    






class CSAR_SimpleCovLayer(nn.Module):
    """
    [CSAR Basic]
    복잡한 기교 제거. 가장 안정적인 기초 버전.
    
    1. Membership: Softmax (0~1, Sum=1) -> 발산 원천 차단
    2. G Matrix: Correlation (Cosine Sim) -> 스케일 폭주 차단
    3. Propagation: EMA G 사용 -> 학습 안정성 확보
    """
    def __init__(self, num_interests, embedding_dim, ema_momentum=0.99, cov_method='cross'):
        super(CSAR_SimpleCovLayer, self).__init__()
        self.K = num_interests
        self.d = embedding_dim
        self.ema_momentum = ema_momentum
        self.cov_method = cov_method # 'cross' or 'union'
        
        # 1. Interest Keys
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        # 2. Global G (EMA Buffer)
        self.register_buffer('ema_G', torch.eye(num_interests))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)
        # 초기화: P의 구조를 G에 복사
        with torch.no_grad():
            P_norm = F.normalize(self.interest_keys, p=2, dim=-1)
            self.ema_G.copy_(torch.matmul(P_norm, P_norm.t()))

    def get_membership(self, embs):
        """
        [Basic 1] Cosine Similarity Membership
        - Softmax Gradient Vanishing 해결
        - Cosine Similarity (-1~1) -> (sim + 1) / 2 -> (0~1)로 변환
        - Intensity 대신 'Directional Alignment' 강조
        """
        # Normalize embeddings and keys
        embs_norm = F.normalize(embs, p=2, dim=-1)
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        # Cosine Similarity: [..., d] @ [K, d].T -> [..., K]
        sim = torch.matmul(embs_norm, keys_norm.t())
        # Map to 0~1 range linearly
        return sim

    @torch.no_grad()
    def update_ema_G(self, user_mem, item_mem):
        """
        [Basic 2] Configurable Correlation EMA
        - cov_method='all': [U; I]^T @ [U; I] (Full Graph)
        - cov_method='cross': (U^T @ I + I^T @ U)/2 (Pure Interaction)
        - cov_method='self': (U^T @ U + I^T @ I)/2 (Self Structure Only)
        """
        # Ensure 2D
        if user_mem.dim() > 2: user_mem = user_mem.view(-1, self.K)
        if item_mem.dim() > 2: item_mem = item_mem.view(-1, self.K)
        
        # 1. Centering (Common for all)
        # 중요: 각자의 평균을 빼서 '변화량'만 남김 -> Negative Correlation 포착
        user_mem = user_mem - user_mem.mean(dim=0, keepdim=True)
        item_mem = item_mem - item_mem.mean(dim=0, keepdim=True)
        
        # 2. Normalize (Column-wise)
        U_norm = F.normalize(user_mem, p=2, dim=0, eps=1e-8)
        I_norm = F.normalize(item_mem, p=2, dim=0, eps=1e-8)
        
        batch_G = None
        
        if self.cov_method == 'all' or self.cov_method == 'union':
            # Union Correlation: [U; I]^T @ [U; I]
            combined = torch.cat([user_mem, item_mem], dim=0)
            C_norm = F.normalize(combined, p=2, dim=0, eps=1e-8)
            batch_G = torch.matmul(C_norm.t(), C_norm)
            
        elif self.cov_method == 'cross':
            # Cross-Correlation Only: Pure Interaction
            # No Identity Addition, No Scaling. Just raw correlation.
            cross_corr = torch.matmul(U_norm.t(), I_norm)
            cross_sym = (cross_corr + cross_corr.t()) / 2.0
            batch_G = cross_sym
            
        elif self.cov_method == 'self':
            # Self-Correlation Only: (U^T U + I^T I) / 2
            u_conn = torch.matmul(U_norm.t(), U_norm)
            i_conn = torch.matmul(I_norm.t(), I_norm)
            batch_G = (u_conn + i_conn) / 2.0
            
        else:
            raise ValueError(f"Unknown cov_method: {self.cov_method}")
        
        # EMA Update
        self.ema_G = self.ema_momentum * self.ema_G + (1 - self.ema_momentum) * batch_G

    def forward(self, u_emb, i_emb=None):
        # 1. Membership
        u_mem = self.get_membership(u_emb)
        
        # update_ema_G is called explicitly in calc_loss with (user + item)
        # so we don't call it here.
        
        # 2. Propagation (Using Stable EMA G)
        # 입력(0~1) @ G(-1~1) -> 출력(안정적)
        u_trans = torch.matmul(u_mem, self.ema_G)
        
        # 3. Scoring
        if i_emb is not None:
            i_mem = self.get_membership(i_emb)
            # 별도의 스케일링 불필요 (값이 작음)
            return torch.sum(u_trans * i_mem, dim=-1)
        else:
            return u_trans

    def get_consistency_loss(self):
        """
        [Basic 3] Structure Alignment
        - P도 Cosine, G도 Cosine -> MSE로 정렬
        - Important: Diagonal Mismatch Problem 해결
          P@P.T의 대각선은 1.0(Normalized). G(Cross)의 대각선은 < 1.0(Interaction Strength).
          억지로 대각선을 맞추려다 붕괴하지 않도록, **대각 성분은 Loss 계산에서 제외(Masking)**함.
          오직 Off-Diagonal(Interaction Pattern)만 학습.
        """
        # Student P
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        G_geo = torch.matmul(keys_norm, keys_norm.t())
        
        # Teacher G
        target = self.ema_G.detach()
        
        # Off-Diagonal Mask
        K = self.K
        mask = 1.0 - torch.eye(K, device=target.device)
        
        # Masked MSE
        loss = F.mse_loss(G_geo * mask, target * mask)
        
        return loss


class CSAR_DistillLayer(CSAR_SimpleCovLayer):
    """
    [CSAR Distill] Intrinsic Kernel + KL Consistency
    - Kernel: P@P.T (Intrinsic Structure) ensures smooth inference.
    - Teacher: EMA_G (Data Statistics) guides P via KL Divergence.
    """
    def __init__(self, num_interests, embedding_dim, ema_momentum=0.99, cov_method='cross', distill_temp=0.1):
        super(CSAR_DistillLayer, self).__init__(num_interests, embedding_dim, ema_momentum, cov_method)
        self.distill_temp = distill_temp

    def forward(self, u_emb, i_emb=None):
        # 1. Membership
        u_mem = self.get_membership(u_emb)
        
        # 2. Propagation (Using Intrinsic Kernel P@P.T)
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        G_intrinsic = torch.matmul(keys_norm, keys_norm.t())
        
        u_trans = torch.matmul(u_mem, G_intrinsic)
        
        # 3. Scoring
        if i_emb is not None:
            i_mem = self.get_membership(i_emb)
            return torch.sum(u_trans * i_mem, dim=-1)
        else:
            return u_trans

    def get_consistency_loss(self):
        """
        KL Divergence with Diagonal Masking
        """
        # Student P
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        G_geo = torch.matmul(keys_norm, keys_norm.t())
        
        # Teacher G
        target = self.ema_G.detach()
        
        # Diagonal Mask for Softmax (Set diagonal to -inf)
        K = self.K
        eye = torch.eye(K, device=target.device)
        mask_val = -1e9
        
        # Apply Mask (Original values + Mask for diagonal)
        # We want to keep off-diagonals as they are, and kill diagonals.
        student_logits = G_geo.masked_fill(eye.bool(), mask_val)
        teacher_logits = target.masked_fill(eye.bool(), mask_val)
        
        # Sharpening
        student_logits = student_logits / self.distill_temp
        teacher_logits = teacher_logits / self.distill_temp
        
        # LogSoftmax (Student) vs Softmax (Teacher)
        log_prob_S = F.log_softmax(student_logits, dim=-1)
        prob_T = F.softmax(teacher_logits, dim=-1)
        
        # KL Divergence
        loss = F.kl_div(log_prob_S, prob_T, reduction='batchmean')
        
        return loss





class CSAR_KLLayer(nn.Module):
    """
    [CSAR KL: Distribution Matching via Deep Clustering]
    
    Mechanism:
    1. Assignment (E-step): P를 기준으로 임베딩을 클러스터링 (No Grad).
    2. Modeling (M-step): 클러스터 중심점(Centroid) 간의 관계를 Target으로 설정.
    3. Alignment: P의 내부 구조(PP^T)가 Target 분포를 따르도록 KL-Div 학습.
    """
    def __init__(self, num_interests, embedding_dim, kl_temp=0.1):
        super(CSAR_KLLayer, self).__init__()
        self.K = num_interests
        self.d = embedding_dim
        self.kl_temp = kl_temp
        
        # Interest Keys (Learnable Prototypes)
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        
        # Scale for Softplus (Intensity Normalization)
        # 차원 수에 따른 내적 값 스케일 보정
        self.scale = nn.Parameter(torch.tensor(embedding_dim ** -0.5))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)

    def get_membership(self, embs):
        """Softplus Membership with Learnable Scale"""
        # einsum으로 배치 차원 유연성 확보 ([B, D], [B, N, D] 모두 가능)
        logits = torch.matmul(embs, self.interest_keys.t()) * self.scale
        return F.softplus(logits)
        
    def get_pp_matrix(self):
        """Intrinsic Structure (Student)"""
        # P 벡터들끼리의 코사인 유사도
        keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
        return torch.matmul(keys_norm, keys_norm.t())

    def get_batch_geometry(self, u_embs, i_embs):
        """
        Calculate Data-Driven Correlation (Teacher)
        """
        # 1. Gather all embeddings
        if u_embs.dim() > 2: u_embs = u_embs.view(-1, self.d)
        if i_embs.dim() > 2: i_embs = i_embs.view(-1, self.d)
        
        all_embs = torch.cat([u_embs, i_embs], dim=0)
        
        # 2. Soft Assignment (No Gradient Here!)
        # "현재 P를 기준으로 이 데이터들은 어떻게 뭉치는가?"
        with torch.no_grad():
            keys_norm = F.normalize(self.interest_keys, p=2, dim=-1)
            embs_norm = F.normalize(all_embs, p=2, dim=-1)
            
            sim = torch.matmul(embs_norm, keys_norm.t())
            
            # Sharpness 조절 (Assignment를 명확하게)
            assignment = F.softmax(sim / 0.1, dim=-1)
            
        # 3. Weighted Centroids
        # 각 관심사(Cluster)의 실제 무게중심 계산
        # [K, B] @ [B, D] -> [K, D]
        centroids = torch.matmul(assignment.t(), all_embs)
        
        # Normalize Centroids (방향만 비교)
        # eps=1e-8: 할당된 데이터가 없는 클러스터 방어
        centroids_norm = F.normalize(centroids, p=2, dim=-1, eps=1e-8)
        
        # 4. Batch Geometry
        # 실제 데이터 클러스터들 간의 각도
        return torch.matmul(centroids_norm, centroids_norm.t())

    def forward(self, u_emb, i_emb=None):
        # 1. Membership
        u_mem = self.get_membership(u_emb)
        
        # 2. Propagation (Using Learned Structure)
        # 외부 통계(G) 없이, 학습된 P 자체의 구조를 사용
        PP = self.get_pp_matrix()
        u_trans = torch.matmul(u_mem, PP)
        
        # 3. Scoring
        if i_emb is not None:
            i_mem = self.get_membership(i_emb)
            return torch.sum(u_trans * i_mem, dim=-1)
        else:
            return u_trans

    def get_consistency_loss(self, u_emb, i_emb):
        """
        KL Divergence: Student(PP) -> Teacher(Data Geometry)
        """
        # Student: PP (Intrinsic)
        PP = self.get_pp_matrix()
        
        # Teacher: Batch Geometry (Data)
        # [중요] Teacher는 정답지이므로 Gradient를 끊어야 함 (.detach())
        G_batch = self.get_batch_geometry(u_emb, i_emb).detach()
        
        # Diagonal Masking
        # 자기 자신과의 유사도(1.0)는 구조 학습에 방해되므로 마스킹
        mask_val = -1e9
        eye = torch.eye(self.K, device=PP.device).bool()
        
        # Scaling & Masking
        student_logits = (PP / self.kl_temp).masked_fill(eye, mask_val)
        teacher_logits = (G_batch / self.kl_temp).masked_fill(eye, mask_val)
        
        # Distribution
        log_prob_S = F.log_softmax(student_logits, dim=-1)
        prob_T = F.softmax(teacher_logits, dim=-1)
        
        # Loss: KL(Teacher || Student)가 아니라 KL(Student || Teacher)
        # 즉, Student가 Teacher의 분포를 닮아가야 함
        loss = F.kl_div(log_prob_S, prob_T, reduction='batchmean')
        
        return loss

class CSAR_KL(nn.Module):
    def __init__(self, num_interests, embedding_dim, kernel_type='partial', pretrained_P=None):
        super().__init__()
        self.K = num_interests
        self.d = embedding_dim
        self.kernel_type = kernel_type
        
        # Learnable Prototypes
        if pretrained_P is not None:
             # SVD based Initialization
             self.P = nn.Parameter(pretrained_P.clone().detach())
        else:
             # Random Initialization
             self.P = nn.Parameter(torch.empty(num_interests, embedding_dim))
             nn.init.xavier_uniform_(self.P)

    def get_mem(self, x):
        """Membership: Cosine Projection (Centered P)"""
        x_norm = F.normalize(x, p=2, dim=-1)
        # Use Centered P for membership? No, membership is usually cosine.
        # But if we view P as correlation, maybe we should center P here too?
        # Let's keep mem as simple cosine for now, but center P for intrinsic structure.
        p_norm = F.normalize(self.P, p=2, dim=-1)
        return torch.matmul(x_norm, p_norm.T)

    def get_membership(self, x):
        """Analysis Helper: Alias for get_mem"""
        return self.get_mem(x)

    def get_intrinsic_correlation(self):
        """Helper: Calculate Pearson Correlation of Prototypes (P)"""
        # Center P along embedding dim (d) -> Mean of each prototype vector
        # This means "how similar are the deviations of prototype vectors from their mean"
        # No, Pearson of "Interests" means we treat each Interest (K) as a variable.
        # The observations are the embedding dimensions (d).
        # So we center along dim=1 (d).
        P_centered = self.P - self.P.mean(dim=1, keepdim=True)
        P_norm = F.normalize(P_centered, p=2, dim=1)
        return torch.matmul(P_norm, P_norm.T)

    def forward(self, u_emb, i_emb=None):
        """
        Inference: Propagation based on Kernel Type
        """
        u_mem = self.get_mem(u_emb)
        
        # 1. Intrinsic Correlation (Pearson of P)
        G_intrinsic = self.get_intrinsic_correlation()
        
        if self.kernel_type == 'partial':
            # --- Partial Correlation (Inverse Precision) ---
            # 2. Precision Matrix Calculation
            K = self.K
            eps = 1e-4
            I = torch.eye(K, device=G_intrinsic.device)
            
            try:
                G_reg = G_intrinsic + eps * I
                Precision = torch.linalg.inv(G_reg)
            except RuntimeError:
                G_reg = G_intrinsic + 1e-2 * I
                Precision = torch.linalg.inv(G_reg)

            # 3. Partial Correlation Transformation
            prec_diag = torch.diag(Precision)
            D_inv_sqrt = torch.diag(torch.pow(prec_diag, -0.5))
            
            G_partial = - torch.matmul(torch.matmul(D_inv_sqrt, Precision), D_inv_sqrt)
            G_partial.fill_diagonal_(1.0)
            
            kernel = G_partial
            
        elif self.kernel_type == 'raw':
            # --- Raw Correlation (Direct Pearson) ---
            # Use G_intrinsic directly as propagation kernel
            # G_intrinsic has 1.0 on diagonal (due to normalization)
            kernel = G_intrinsic
            
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")
        
        # 4. Propagation
        u_trans = torch.matmul(u_mem, kernel)
        
        if i_emb is not None:
            i_mem = self.get_mem(i_emb)
            return (u_trans * i_mem).sum(dim=-1)
        return u_trans

    def get_gram_matrix(self):
        """Analysis Helper: Intrinsic Structure (Pearson)"""
        return self.get_intrinsic_correlation()

    def get_loss(self, u_emb, i_emb):
        """
        Training: Align P (Pearson) with 'Real Interaction Pearson Correlation' using MSE
        """
        # ---------- 1. Teacher: Real Interaction (Pearson Correlation) ----------
        with torch.no_grad():
            if u_emb.dim() > 2: u_emb = u_emb.view(-1, self.d)
            if i_emb.dim() > 2: i_emb = i_emb.view(-1, self.d)
            
            u_mem = self.get_mem(u_emb) # [B, K]
            i_mem = self.get_mem(i_emb) # [B, K]

            # Centering (각 관심사의 평균 제거 -> 인기도 편향 제거)
            u_mem = u_mem - u_mem.mean(dim=0, keepdim=True)
            i_mem = i_mem - i_mem.mean(dim=0, keepdim=True)
            
            # Normalize (Correlation Scale: -1 ~ 1)
            u_norm = F.normalize(u_mem, p=2, dim=0, eps=1e-8)
            i_norm = F.normalize(i_mem, p=2, dim=0, eps=1e-8)
            
            # Cross-Correlation Matrix: (K, B) @ (B, K) -> (K, K)
            # "실제 데이터에서는 유저의 관심사가 아이템의 관심사와 이렇게 연결되더라"
            G_cross = torch.matmul(u_norm.T, i_norm)
            
            # Symmetrization (Target must be symmetric)
            G_target = (G_cross + G_cross.T) / 2.0

        # ---------- 2. Student: Intrinsic Correlation (Pearson) ----------
        G_pred = self.get_intrinsic_correlation()

        # ---------- 3. MSE Loss ----------
        # 대각 성분(Self-loop)은 P의 정규화로 인해 항상 1이므로 학습에서 제외
        # 오직 "관심사 간의 관계(Structure)"만 학습
        K = self.K
        mask = ~torch.eye(K, device=self.P.device, dtype=torch.bool)
        
        # "내 구조(G_pred)가 실제 데이터의 구조(G_target)와 같아지도록"
        return F.mse_loss(G_pred[mask], G_target[mask])

class CSAR_DualProto_Layer(nn.Module):
    """
    [CSAR Dual-Proto: Dual-View Causal Alignment]
    
    Inference (Forward):
    - Symmetrize(P_u @ P_i.T) -> Precision Matrix -> Partial Correlation.
    - "추론 시에는 유저와 아이템이 합의된(Shared) 구조를 통해 전파."
    
    Training (Loss):
    - P_u @ P_i.T -> Match Raw User-Item Interaction (U^T @ I).
    - "학습 시에는 있는 그대로의 비대칭적 상호작용(Raw Truth)을 학습."
    """
    def __init__(self, num_interests, embedding_dim):
        super().__init__()
        self.K = num_interests
        self.d = embedding_dim
        
        # P_u, P_i passed from parent model

    def get_mem(self, x, P):
        """Membership: Cosine Projection"""
        x_norm = F.normalize(x, p=2, dim=-1)
        p_norm = F.normalize(P, p=2, dim=-1)
        return torch.matmul(x_norm, p_norm.T)

    def get_intrinsic_correlation(self, P1, P2=None):
        """Helper: Pearson Correlation of Prototypes"""
        if P2 is None: P2 = P1
        
        # Centering (Pure Direction)
        P1_centered = P1 - P1.mean(dim=1, keepdim=True)
        P2_centered = P2 - P2.mean(dim=1, keepdim=True)
        
        P1_norm = F.normalize(P1_centered, p=2, dim=1)
        P2_norm = F.normalize(P2_centered, p=2, dim=1)
        
        return torch.matmul(P1_norm, P2_norm.T)

    def forward(self, u_emb, P_u, P_i, i_emb=None):
        """
        Inference: Causal Propagation via Consensus Structure
        """
        u_mem = self.get_mem(u_emb, P_u)
        
        # 1. Raw Cross-Interaction (User View vs Item View)
        G_raw = self.get_intrinsic_correlation(P_u, P_i)
        
        # 2. Consensus Symmetrization (The User's Assumption)
        G_shared = (G_raw + G_raw.T) / 2.0
        
        # 3. Precision Matrix (Inverse) -> Partial Correlation
        K = self.K
        eps = 1e-3 # Increased epsilon for stability
        I = torch.eye(K, device=G_shared.device)
        
        try:
            # Try Standard Inverse with strong regularization
            G_reg = G_shared + 1e-2 * I 
            Precision = torch.linalg.inv(G_reg)
        except RuntimeError:
            # Fallback to Pseudo-Inverse which is robust to singularity
            Precision = torch.linalg.pinv(G_shared + 1e-2 * I)

        # 4. Convert to Partial Correlation
        # Formula: rho_ij = - p_ij / sqrt(p_ii * p_jj)
        prec_diag = torch.diag(Precision)
        
        # CRITICAL FIX: Ensure diagonal is positive for sqrt
        prec_diag = torch.clamp(prec_diag, min=1e-6)
        
        D_inv_sqrt = torch.diag(torch.pow(prec_diag, -0.5))
        
        G_partial = - torch.matmul(torch.matmul(D_inv_sqrt, Precision), D_inv_sqrt)
        G_partial.fill_diagonal_(1.0) # Self-loop 복구
        
        # Safety: NaN/Inf Handling
        if torch.isnan(G_partial).any() or torch.isinf(G_partial).any():
            G_partial = torch.nan_to_num(G_partial, nan=0.0, posinf=1.0, neginf=-1.0)
            G_partial.fill_diagonal_(1.0)

        # 5. Propagation
        u_trans = torch.matmul(u_mem, G_partial)
        
        if i_emb is not None:
            i_mem = self.get_mem(i_emb, P_i)
            return (u_trans * i_mem).sum(dim=-1)
        return u_trans

    def get_loss(self, u_emb, i_emb, P_u, P_i):
        """
        Training: Raw Structure Alignment
        """
        # --- Helpers ---
        def get_batch_corr(emb, P):
            if emb.dim() > 2: emb = emb.view(-1, self.d)
            mem = self.get_mem(emb, P)
            mem = mem - mem.mean(dim=0, keepdim=True) 
            mem = F.normalize(mem, p=2, dim=0, eps=1e-8)
            return torch.matmul(mem.T, mem)

        def get_batch_cross_corr(u_e, i_e, P_u, P_i):
             if u_e.dim() > 2: u_e = u_e.view(-1, self.d)
             if i_e.dim() > 2: i_e = i_e.view(-1, self.d)
             u_m = self.get_mem(u_e, P_u)
             i_m = self.get_mem(i_e, P_i)
             u_m = u_m - u_m.mean(dim=0, keepdim=True)
             i_m = i_m - i_m.mean(dim=0, keepdim=True)
             u_n = F.normalize(u_m, p=2, dim=0, eps=1e-8)
             i_n = F.normalize(i_m, p=2, dim=0, eps=1e-8)
             # Raw Cross-Correlation (Not Symmetric)
             return torch.matmul(u_n.T, i_n)
        # ---------------

        # 1. User Internal Structure (Auto-Consistency)
        G_u_target = get_batch_corr(u_emb, P_u).detach()
        G_u_pred = self.get_intrinsic_correlation(P_u)
        loss_u = F.mse_loss(G_u_pred, G_u_target)
        
        # 2. Item Internal Structure (Auto-Consistency)
        G_i_target = get_batch_corr(i_emb, P_i).detach()
        G_i_pred = self.get_intrinsic_correlation(P_i)
        loss_i = F.mse_loss(G_i_pred, G_i_target)
        
        # 3. Cross Interaction Structure (Interaction Consistency)
        # [수정] 대칭화 하지 않음. Raw vs Raw 매칭.
        # Teacher: 실제 배치의 U-I 연결 패턴 (Asymmetric)
        G_cross_target = get_batch_cross_corr(u_emb, i_emb, P_u, P_i).detach()
        
        # Student: P_u와 P_i의 내재적 연결 패턴 (Asymmetric)
        G_cross_pred = self.get_intrinsic_correlation(P_u, P_i)
        
        # "P_u와 P_i야, 너네끼리의 각도가 실제 유저-아이템의 연결 각도와 똑같아져라."
        loss_cross = F.mse_loss(G_cross_pred, G_cross_target)
        
        return loss_u, loss_i, loss_cross

class CSAR_SVD_Layer(CSAR_KL):
    """
    CSAR SVD Layer
    - Propagation: Intrinsic Correlation (P @ P.T) or Partial depending on kernel_type
    - Consistency Loss: Align P @ P.T with FIXED SVD Structure (V @ V.T)
    """
    def __init__(self, num_interests, embedding_dim, kernel_type='partial', fixed_G=None):
        super().__init__(num_interests, embedding_dim, kernel_type)
        
        # Fixed Target Structure (V @ V.T)
        if fixed_G is not None:
            self.register_buffer('fixed_G', fixed_G)
        else:
            self.fixed_G = None

    def get_loss(self, u_emb=None, i_emb=None):
        """
        Training: Align Intrinsic Structure (P) to Fixed SVD Structure (V)
        """
        if self.fixed_G is None:
            return torch.tensor(0.0)
            
        # Student: Intrinsic Correlation (Pearson of P)
        G_pred = self.get_intrinsic_correlation()
        
        # Teacher: Fixed SVD Structure
        G_target = self.fixed_G
        
        # MSE Loss
        # We want P structure to adhere to the global co-occurrence structure.
        return F.mse_loss(G_pred, G_target)