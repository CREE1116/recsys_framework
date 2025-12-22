import torch
import torch.nn as nn
import numpy as np  
import torch.nn.functional as F
import math

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) 손실 함수
    """
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

class DynamicMarginBPRLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(DynamicMarginBPRLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pos_scores, neg_scores):
        # 1. 현재 모델이 느끼는 난이도 (점수 차이)
        with torch.no_grad():
            diff = pos_scores - neg_scores
            # 차이가 클수록(잘할수록) 마진을 세게 검
            dynamic_margin = self.alpha * torch.sigmoid(diff) 
        # 2. 동적 마진 적용
        loss = F.softplus(-(pos_scores - neg_scores - dynamic_margin)).mean()
        return loss

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        """
        epsilon: 
          > 0 : 쉬운 문제도 더 파고들어라! (희소 데이터 추천, Gradient 강화)
          < 0 : 어려운 문제에 집중해라! (Focal Loss 느낌)
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        # 1. 기본 CE Loss 계산 (Base)
        ce_loss = self.ce(logits, targets)
        
        # 2. 정답 클래스의 확률(Pt) 계산
        # Softmax를 통과시킨 후 정답 인덱스의 확률만 가져옴
        pt = torch.gather(F.softmax(logits, dim=-1), 1, targets.unsqueeze(1)).squeeze(1)
        
        # 3. PolyLoss 수식 적용
        # Loss = CE + eps * (1 - Pt)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss

class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) 손실 함수
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predictions, ratings):
        loss = self.loss_fn(predictions, ratings)
        return loss
    

class scaled_sampled_softmax(nn.Module):

    def __init__(self, temperature=0.4, scale_factor=20.0):
        super(scaled_sampled_softmax, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, pos_scores, neg_scores):
        pos_scores = pos_scores / self.temperature  # (B,)
        neg_scores = neg_scores / self.temperature  # (B, N_neg)

        # scale_factor를 tensor로 만들어 autograd 연결 유지
        scale_factor = torch.tensor(self.scale_factor, device=pos_scores.device, dtype=pos_scores.dtype)
        log_scale = torch.log(scale_factor)  # scalar tensor, requires_grad=False
        # log(C * exp(neg/T)) = log(C) + neg/T
        neg_terms = log_scale + neg_scores  # (B, N_neg)
        # (B, 1 + N_neg)
        all_terms = torch.cat([pos_scores.unsqueeze(1), neg_terms], dim=1)
        # log(exp(pos/T) + C * sum(exp(neg/T)))
        log_denominator = torch.logsumexp(all_terms, dim=1)  # (B,)
        # -log(P(pos)) = log_denominator - pos/T
        loss = log_denominator - pos_scores
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE 손실 함수
    """
    def __init__(self, temperature = 0.4):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores (torch.Tensor): (batch_size, 1)
            neg_scores (torch.Tensor): (batch_size, num_negatives)
        """
        # (batch_size, 1 + num_negatives)
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        all_scores = all_scores / self.temperature
        
        # Positive score에 해당하는 인덱스는 항상 0
        labels = torch.zeros(all_scores.shape[0], dtype=torch.long, device=all_scores.device)
        
        loss = F.cross_entropy(all_scores, labels)
        return loss

class NormalizedSampledSoftmaxLoss(nn.Module):
    """
    Normalized Sampled Softmax Loss
    
    특징:
    1. Z-Score Normalization: 입력된 에너지 점수의 분포를 정규화하여(Mean=0, Std=1) 학습 안정성을 극대화함.
       - Temperature 의존성 문제를 해결하고, Gradient Explosion을 방지.
    2. Sampled Softmax Correction: In-Batch Negative Sampling으로 인한 편향을 보정.
       - Correction Term: log((N-1)/(B-1))을 Negative Logits에 더해줌으로써, 
         마치 전체 아이템(N)에 대해 Softmax를 수행하는 것과 유사한 효과를 냄.
    3. Fixed Temperature: 정규화된 분포에 맞춰 고정된 온도를 사용하여 학습의 일관성 유지.
    """
    def __init__(self, n_items, temperature=0.1):
        """
        Args:
            n_items (int): 전체 아이템 수 (Sampled Softmax 보정용)
            temperature (float): 고정된 온도 값.
        """
        super(NormalizedSampledSoftmaxLoss, self).__init__()
        
        self.n_items = n_items
        self.temperature = temperature

    def forward(self, scores, is_explicit=False):
        """
        Args:
            scores (Tensor): 
                - If is_explicit=False: [Batch, Batch] (User x Item Matrix)
                - If is_explicit=True:  [Batch, 1 + NumNegatives] (Column 0 is Pos)
            is_explicit (bool): Whether input scores use explicit negative sampling.
        """
        # 1. Energy Score Calculation (Removed internal matmul)
        energy_scores = scores
        
        # 2. Global Z-Score Normalization removed

        
        # 3. Fixed Temperature Scaling
        logits = energy_scores / self.temperature
        
        # 4. Sampled Softmax Correction (Consistent Estimation)
        # In-Batch Negative들이 전체 Negative의 일부임을 감안하여 점수 보정
        # 보정: Neg Logits += log((N-1)/(B-1))
        # [Condition] Correction is ONLY for In-Batch Sampling (Implicit)
        if self.training and not is_explicit:
            batch_size = logits.size(0)
            # 정확한 추정을 위해 (N-1)/(B-1) 사용 (User 본인의 Pos 제외)
            correction = math.log((self.n_items - 1) / (batch_size - 1 + 1e-9))
            
            # Diagonal(Pos)은 건드리지 않고, Off-Diagonal(Neg)에만 보정항 추가
            # Mask 생성: Identity Matrix (1 on diagonal)
            mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
            
            # Negatives(False in mask) apply correction
            # In-place add to save memory
            logits = logits + (~mask) * correction
        
        # 5. Cross Entropy
        if is_explicit:
            # Explicit: Column 0 is always Positive
            labels = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
        else:
            # In-Batch: Diagonal is Positive
            labels = torch.arange(scores.size(0), device=scores.device)
            
        loss = F.cross_entropy(logits, labels)
        
        return loss


class NDCGWeightedListwiseBPR(nn.Module):
    """
    Listwise BPR with NDCG-style position weighting
    (Optimized version)
    """
    def __init__(self, k=10, use_zscore=False, is_explicit=False):
        super().__init__()
        self.k = k
        self.use_zscore = use_zscore
        self.is_explicit = is_explicit
    
    def forward(self, scores):
        """
        Args:
            scores (Tensor): [Batch, Batch] - 점수 행렬 (User x Item)
        """
        # 0. Score matrix (Taken as input)
        # Apply Z-score Normalization if requested
        if self.use_zscore:
            mean = scores.mean()
            std = scores.std()
            scores = (scores - mean) / (std + 1e-9)

        B = scores.size(0)
        device = scores.device
        
        if self.is_explicit:
            # Case 1: Explicit Negative Sampling input [B, 1 + NumNeg]
            # Column 0 is Positive. Columns 1..N are Negatives.
            pos_scores = scores[:, 0].unsqueeze(1) # [B, 1]
            neg_scores = scores[:, 1:]             # [B, N]
            
            # Pairwise differences
            diff = pos_scores - neg_scores         # [B, N]
            
            # Ranking: Sort all scores (Pos + Negs) row-wise
            # We want to know the rank of items.
            # But wait, weights depend on rank.
            # Ranks are calculated over (1 + N) items.
            sorted_indices = scores.argsort(dim=1, descending=True)
            ranks = torch.empty_like(scores)
            ranks.scatter_(
                dim=1,
                index=sorted_indices,
                src=torch.arange(scores.size(1), device=device, dtype=scores.dtype).view(1, -1).expand(B, -1)
            )
            
            # We only care about weights for the Negatives (Columns 1..N)
            neg_ranks = ranks[:, 1:] # [B, N]
            
            # NDCG Weights for Negatives
            # Note: The weight depends on WHERE the negative is ranked.
            # If Rank is 0 (it beat the positive), weight is high.
            ndcg_weights = 1.0 / torch.log2(neg_ranks + 2.0)
            
            # Top-K Masking (on Negatives)
            # If negative rank is >= K, mask it out.
            topk_mask = neg_ranks < self.k
            
            # Final Mask
            final_mask = topk_mask.float()
            
            # Loss
            # bpr_term = -log(sigmoid(pos - neg))
            bpr_term = -torch.log(torch.sigmoid(diff).clamp(min=1e-8))
            weighted_loss = bpr_term * ndcg_weights * final_mask
            
            return weighted_loss.sum() / final_mask.sum().clamp(min=1.0)

        else:
            # Case 2: Implicit / In-Batch (Square Matrix [B, B])
            # Diagonal is Positive.
            labels = torch.arange(B, device=device)
            pos_scores = scores[labels, labels].unsqueeze(1)  # [B, 1]
            
            # Pairwise differences
            diff = pos_scores - scores  # [B, B]
            
            # 4. Efficient ranking computation
            # Method 1: Direct argsort (current)
            sorted_indices = scores.argsort(dim=1, descending=True)
            ranks = torch.empty_like(scores)
            ranks.scatter_(
                dim=1,
                index=sorted_indices,
                src=torch.arange(B, device=device, dtype=scores.dtype).view(1, -1).expand(B, -1)
            )
            
            # 5. NDCG discount weights
            ndcg_weights = 1.0 / torch.log2(ranks + 2.0)  # [B, B]
            
            # 6. Masking
            # Self-pairs 제외
            self_mask = ~torch.eye(B, device=device, dtype=torch.bool)
            
            # Top-K만
            topk_mask = ranks < self.k
            
            # Optional: Violation만 (negative > positive)
            # User Feedback: "Remove only_violation entirely, allow learning from all Top-K items"
            # This enforces a margin even for easy negatives.
            final_mask = (self_mask & topk_mask).float()
            
            # 7. Weighted BPR Loss
            bpr_term = -torch.log(torch.sigmoid(diff).clamp(min=1e-8))
            weighted_loss = bpr_term * ndcg_weights * final_mask
            
            # 8. Normalization
            return weighted_loss.sum() / final_mask.sum().clamp(min=1.0)

class TopK_NDCG_BPR(nn.Module):
    def __init__(self, k=20, only_violation=False):
        super().__init__()
        self.k = k
        self.only_violation = only_violation
        
    def forward(self, scores):
        """
        scores: [Batch, Batch] (User x Item Matrix)
        """
        B = scores.size(0)
        device = scores.device
        
        # 1. Positive Score (대각선) 추출 [B, 1]
        # 내 정답 점수는 미리 챙겨둡니다.
        labels = torch.arange(B, device=device)
        pos_scores = scores[labels, labels].unsqueeze(1)
        
        # 2. Top-(K+1) 추출 (Efficient Hard Negative Mining)
        # 내 정답이 Top-K 안에 포함될 수도 있으니 넉넉하게 K+1개를 뽑습니다.
        # 이렇게 하면 전체 정렬(Argsort) 없이 상위권 놈들만 딱 데려옵니다.
        # topk_vals: [B, K+1], topk_inds: [B, K+1]
        k_val = min(self.k + 1, B)
        topk_vals, topk_inds = torch.topk(scores, k=k_val, dim=1)
        
        # 3. 가중치 계산 (LambdaRank Style)
        # 뽑힌 애들은 무조건 0등, 1등, 2등... 순서대로입니다.
        # 따라서 랭킹 계산을 따로 할 필요 없이 그냥 arange로 주면 됩니다.
        ranks = torch.arange(k_val, device=device).unsqueeze(0).expand(B, -1)
        weights = 1.0 / torch.log2(ranks + 2.0) # [B, K+1]
        
        # 4. 비교 (Positive vs Top-K Negatives)
        # [B, 1] vs [B, K+1] -> Broadcasting -> [B, K+1]
        # 내 정답 점수와, 상위권 랭커들의 점수를 비교합니다.
        diff = pos_scores - topk_vals
        
        # 5. 마스킹 (자기 자신 제외)
        # Top-K 안에 내 정답(Positive)이 섞여 있을 수 있습니다. 걔는 Loss에서 빼야 합니다.
        # topk_inds(후보 인덱스)가 labels(내 인덱스)와 같으면 True
        is_self = (topk_inds == labels.unsqueeze(1))
        
        # 기본 마스크: 자기 자신이 아닌 것들만 살림
        mask = (~is_self).float()
        
        # (옵션) Violation 마스크: 나보다 점수 높은 놈만 살림
        if self.only_violation:
            mask = mask * (topk_vals > pos_scores).float()
            
        # 6. 최종 Weighted BPR Loss
        # 상위권 놈들에 대해 BPR을 구하고, 순위 가중치를 곱합니다.
        loss_map = -F.logsigmoid(diff) * weights * mask
        
        # 7. 평균 내서 리턴
        # 유효한 샘플 수로 나눠줍니다 (division by zero 방지)
        return loss_map.sum() / mask.sum().clamp(min=1.0)


class InBatchHardBPR(nn.Module):
    """
    Stable In-Batch BPR:
    - pos: diagonal
    - neg: batch의 모든 item
    - only hard negatives
    - no rank, no ndcg, no top-k
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores):
        """
        scores: [B, B]  (i-th row: scores of user i over items in batch)
        diagonal entry = positive
        """
        B = scores.size(0)
        device = scores.device
        
        # 1. positive
        labels = torch.arange(B, device=device)
        pos = scores[labels, labels].unsqueeze(1)   # [B, 1]
        
        # 2. all negatives
        neg = scores                                # [B, B]

        # 3. diff = pos - neg (broadcasting)
        diff = (pos - neg) / self.temperature       # [B, B]
        
        # 4. mask self-comparison
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        
        # 5. hard negatives: neg > pos
        hard_mask = (neg > pos).detach()
        
        mask = self_mask & hard_mask                # [B, B]
        mask = mask.float()
        
        # 6. BPR loss
        loss_map = -F.logsigmoid(diff) * mask
        
        # 7. normalization
        return loss_map.sum() / mask.sum().clamp(min=1.0)


class EntropyAdaptiveInfoNCE(nn.Module):
    """
    SCI 논문용: Decoupled Adaptive InfoNCE
    - get_adaptive_tau: 학습 상태(Entropy, Orthogonality)를 진단하여 온도를 결정 (Gradient 차단됨)
    - forward: 결정된 온도로 실제 Loss 계산 (Unidirectional for efficiency)
    """
    def __init__(self, base_tau=0.1, alpha=0.5, eps=1e-8):
        super().__init__()
        self.base_tau = base_tau
        self.alpha = alpha
        self.eps = eps

    def _calc_uncertainty(self, z):
        """Interest Vector의 불확실성(Entropy) 측정"""
        B, K = z.size()
        p = F.softmax(z, dim=1)
        log_p = F.log_softmax(z, dim=1)
        entropy = -(p * log_p).sum(dim=1)
        max_entropy = math.log(K + self.eps)
        return entropy / max_entropy # [B]

    def _calc_key_quality(self, W):
        """Key의 직교성 품질 측정 (1.0 = 완벽)"""
        K = W.size(0)
        W_norm = F.normalize(W, dim=1)
        gram = W_norm @ W_norm.T
        eye = torch.eye(K, device=W.device)
        # 비대각 원소의 평균 크기
        off_diag_mean = (gram - eye).abs().sum() / (K * (K - 1) + self.eps)
        return 1.0 - off_diag_mean

    def get_adaptive_tau(self, interests, W):
        """
        외부에서 호출: 현재 배치의 Interest와 Key 상태를 보고 Tau를 반환
        """
        # 1. Key Quality (Global) - 안전장치: no_grad
        with torch.no_grad():
            key_quality = self._calc_key_quality(W)
        
        # 2. Instance Uncertainty (Local)
        # interest 자체는 gradient가 필요할 수 있으나, tau 계산 목적으론 끊는게 안전함
        with torch.no_grad(): 
            uncertainty = self._calc_uncertainty(interests)
            
            # 3. Decoupled Adaptation Logic
            # Uncertainty 높음(Sparse) -> Tau 증가 (Soft)
            # Quality 높음(Orthogonal) -> Tau 감소 (Sharp)
            net_difficulty = uncertainty - key_quality
            
            tau = self.base_tau * torch.exp(self.alpha * net_difficulty)
            tau = torch.clamp(tau, min=0.05, max=0.5).unsqueeze(1) # [B, 1]
        
        return tau

    def forward(self, view1, view2, tau):
        """
        Unidirectional InfoNCE
        - view1, view2: [B, K] (Augmented Views)
        - tau: [B, 1] (Adaptive Temperature)
        """
        B = view1.size(0)
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        
        # Cosine Similarity / Adaptive Tau
        logits = (z1 @ z2.T) / tau
        
        # Labels: Diagonal is positive
        labels = torch.arange(B, device=z1.device)
        
        # One-way Cross Entropy (속도 2배, 메모리 절약)
        return F.cross_entropy(logits, labels)



class orthogonal_loss(nn.Module):
    """
    직교성(Orthogonality) 손실 함수
    """
    def __init__(self, losstype):
        super(orthogonal_loss, self).__init__()
        self.loss_type = losstype

    def forward(self, keys):
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        num_off_diagonal_elements = K * K - K
        if self.loss_type == 'l1':
            identity_matrix = torch.eye(keys.size(0), device=keys.device)
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보    
            loss = torch.abs(cosine_similarity - identity_matrix).sum()
            return loss / num_off_diagonal_elements 
        elif self.loss_type == 'l2':
            l2_sum_loss = ((cosine_similarity - identity_matrix) ** 2).sum()
            return l2_sum_loss / num_off_diagonal_elements 
        
    def l1_orthogonal_loss(keys):
        """
        Enforces orthogonality on the interest keys.

        Args:
            keys (torch.Tensor): The global interest keys tensor. [num_interests, D]

        Returns:
            torch.Tensor: The orthogonality loss.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        num_off_diagonal_elements = K * K - K
        identity_matrix = torch.eye(keys.size(0), device=keys.device)
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보
        loss = torch.abs(cosine_similarity - identity_matrix).sum()
        return loss / num_off_diagonal_elements 

    def l2_orthogonal_loss(keys):
        """
        Enforces orthogonality on the interest keys.

        Args:
            keys (torch.Tensor): The global interest keys tensor. [num_interests, D]

        Returns:
            torch.Tensor: The orthogonality loss.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        off_diag = cosine_similarity - identity_matrix
            # 1. 제곱의 합 (L2 Loss)
        l2_sum_loss = (off_diag ** 2).sum()
            # 2. 비대각선 요소의 개수로 나누어 정규화 (Normalization)
        num_off_diagonal_elements = K * K - K
        normalized_loss = l2_sum_loss / num_off_diagonal_elements 
        return normalized_loss 