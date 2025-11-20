import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoSupportAttentionLayer(nn.Module):
    """
    d-차원의 임베딩을 K-차원의 비음수 관심사 가중치 벡터로 변환하는 레이어.
    """
    def __init__(self, num_interests, embedding_dim, scale=False):
        super(CoSupportAttentionLayer, self).__init__() 
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        # K-Anchor (관심사 키)
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(self.num_interests**-0.5)) if scale else torch.tensor(1.0)
        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 초기화를 사용하여 관심사 키를 초기화합니다."""
        nn.init.xavier_uniform_(self.interest_keys)

    def forward(self, embedding_tensor):
        """
        입력 텐서를 K-차원 관심사 가중치로 변환합니다.
        
        Args:
            embedding_tensor (torch.Tensor): [..., d] shape의 임베딩 텐서.
        
        Returns:
            torch.Tensor: [..., K] shape의 비음수 관심사 가중치 텐서.
        """
        # scale이 Parameter인 경우와 Tensor인 경우 모두 처리
        scale_val = self.scale if isinstance(self.scale, torch.Tensor) else 1.0
        
        attention_logits = torch.einsum('...d,kd->...k', embedding_tensor, self.interest_keys) * scale_val
        interest_weights = F.softplus(attention_logits)
        
        return interest_weights
    
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
        if loss_type == "l1":
            return self.l1_orthogonal_loss(self.interest_keys)
        else:
            return self.l2_orthogonal_loss(self.interest_keys)


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
