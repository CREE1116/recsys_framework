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

class ScalableBPRLoss(nn.Module):
    """
    Auto Scaled Margin BPR (ScalableBPRLoss)
    1. Gain: EMA로 입력 점수 분산을 추적하여 '자동 스케일링' (Gradient Free)
    2. Margin: Softplus로 양수 제약 + Adversarial Reg로 '자동 마진 성장'
    """
    def __init__(self, 
                 decay=0.99,          # EMA 반영 속도 (클수록 천천히 변함)
                 target_magnitude=2.0, # 목표로 하는 점수 차이의 절대 크기 (Sigmoid 기울기 최대 구간)
                 init_margin=0.0,
                 margin_weight=0.1):   # 마진을 키우려는 힘의 세기
        super(ScalableBPRLoss, self).__init__()
        
        # [Gain Control] 학습 파라미터 아님 (Buffer)
        self.decay = decay
        self.target_magnitude = target_magnitude
        self.register_buffer('running_gain', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(0.0))

        # [Margin Control] 학습 파라미터 (Softplus 통과 전의 raw 값)
        # 초기값을 역산해서 설정
        # init_margin=0.0 -> softplus(x)=0 -> x = -inf (근사치 -5.0)
        self.raw_margin = nn.Parameter(torch.tensor(-5.0 if init_margin <= 0 else torch.log(torch.exp(torch.tensor(init_margin)) - 1)))
        self.margin_weight = margin_weight

    def get_margin(self):
        return F.softplus(self.raw_margin).item()
        
    def get_gain(self):
        return self.running_gain.item()

    def forward(self, pos_scores, neg_scores):
        # 1. 점수 차이 (Raw Diff)
        # pos_scores: [B] or [B, 1], neg_scores: [B] or [B, K]
        diff = pos_scores - neg_scores
        
        # -----------------------------------------------------------
        # [Step 1] Gain의 EMA 자동 제어 (Training Mode Only)
        # -----------------------------------------------------------
        if self.training:
            with torch.no_grad():
                # 현재 배치의 표준편차 (신호의 세기)
                current_std = torch.std(diff).clamp(min=1e-6)
                
                # 목표: diff * gain ≈ target_magnitude
                # Sigmoid가 가장 민감하게 반응하는 구간(±2~3)으로 신호를 맞춰줌
                ideal_gain = self.target_magnitude / current_std
                
                if self.initialized.item() == 0:
                    self.running_gain.data.copy_(ideal_gain)
                    self.initialized.data.fill_(1.0)
                else:
                    # 급격한 변화를 막기 위해 EMA 적용
                    new_gain = (self.decay * self.running_gain) + ((1 - self.decay) * ideal_gain)
                    self.running_gain.data.copy_(new_gain)
        
        # -----------------------------------------------------------
        
        # 2. Margin의 양수화 (Softplus)
        actual_margin = F.softplus(self.raw_margin)
        
        # 3. 로스 계산
        # Gain(자동) * (Diff - Margin(자동))
        # running_gain은 상수 취급되므로 여기서 Gradient 끊김 (안정성 확보)
        scaled_logits = (self.running_gain + 1e-2) * (diff - actual_margin)
        
        bpr_loss = -F.logsigmoid(scaled_logits).mean()
        
        # 4. 마진 성장 유도 (Adversarial Regularization)
        # Loss를 줄이려면 margin을 줄여야 하는데,
        # Reg Loss는 margin을 키우라고 강요함.
        # -> 두 힘이 평형을 이루는 지점(Auto Margin)까지 마진이 자라남.
        reg_loss = -self.margin_weight * actual_margin
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss

class DynamicMarginBPRLoss(nn.Module):
    """
    동적 마진 BPR 손실 함수
    모델이 쉽게 구분하는 쌍에는 더 큰 마진을 요구
    """
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

class HardAwareBPRLoss(nn.Module):
    """
    어려운 예제(Hard Negative)에 더 집중하는 손실 함수
    """
    def __init__(self, alpha=0.5):
        super(HardAwareBPRLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pos_scores, neg_scores):
        with torch.no_grad():
            diff = pos_scores - neg_scores
            p = torch.sigmoid(diff)
            # 모델이 헷갈릴수록(p가 0.5 이하일수록) 마진을 키움
            # (1-p) 항을 이용해 '못 맞출수록 가중치 부여'
            dynamic_margin = self.alpha * (1 - p)
            
        # Hard sample일수록 margin이 커져서 Loss가 급증함 -> Gradient 폭발적 증가
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
    

class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax 손실 함수
    
    특징:
    1. Temperature Scaling: 점수 분포 조절
    2. Scale Factor: Negative에 가중치 적용하여 Hard Negative 효과 강화 (1.0이면 표준 Softmax)
    3. logQ Correction (선택): Popularity 기반 Negative Sampling 시 인기편향 보정
       - 보정 공식: score - log(Q(i)), Q(i) = 아이템 i의 샘플링 확률
    """
    def __init__(self, temperature=0.1, scale_factor=1.0, log_q_correction=False):
        """
        Args:
            temperature (float): 온도 파라미터 (낮을수록 sharp한 분포)
            scale_factor (float): Negative 가중치 (1.0이면 표준 Softmax, >1이면 Hard Negative 강화)
            log_q_correction (bool): 인기편향 보정 사용 여부
        """
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.log_q_correction = log_q_correction
        self.log_q = None

    def set_log_q(self, sampling_weights):
        """
        Popularity 기반 샘플링 가중치로부터 log(Q(i)) 계산
        
        Args:
            sampling_weights (Tensor): 아이템별 샘플링 확률 [n_items]
        """
        if sampling_weights is not None:
            self.log_q = torch.log(sampling_weights + 1e-10)
        else:
            self.log_q = None

    def forward(self, pos_scores, neg_scores, neg_item_ids=None):
        """
        Args:
            pos_scores: [Batch, 1] Positive 점수
            neg_scores: [Batch, K] Negative 점수
            neg_item_ids: [Batch, K] Negative 아이템 ID (logQ correction 시 필요)
        """
        pos_logits = pos_scores / self.temperature
        neg_logits = neg_scores / self.temperature
        
        # logQ Correction (인기편향 보정)
        if self.log_q_correction and self.log_q is not None and neg_item_ids is not None:
            log_q_neg = self.log_q[neg_item_ids]
            neg_logits = neg_logits - log_q_neg
        
        # Scale Factor 적용 (Hard Negative 강화)
        if self.scale_factor != 1.0:
            log_scale = torch.log(torch.tensor(self.scale_factor, device=pos_logits.device, dtype=pos_logits.dtype))
            neg_logits = log_scale + neg_logits
        
        # [Batch, 1 + K]
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        # Positive는 항상 Column 0
        labels = torch.zeros(all_logits.shape[0], dtype=torch.long, device=all_logits.device)
        
        return F.cross_entropy(all_logits, labels)


class PureContrastiveLoss(nn.Module):
    """
    Pure Contrastive Loss (DCL 스타일)
    
    InfoNCE/SampledSoftmax의 분모에서 Positive를 제외하여,
    순수하게 "Negative와의 대비"만 학습.
    
    장점:
    - 분모에서 자기 자신과 비교하는 비효율 제거
    - Negative 구분에 집중
    
    수식:
    L = -log(exp(s_pos/τ) / Σexp(s_neg/τ))
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): 온도 파라미터 (낮을수록 sharp한 분포)
        """
        super(PureContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [Batch, 1] Positive 점수
            neg_scores: [Batch, K] Negative 점수
        """
        pos_logits = pos_scores / self.temperature  # [B, 1]
        neg_logits = neg_scores / self.temperature  # [B, K]
        
        # 분모: Negative만 (Positive 제외!)
        neg_log_sum_exp = torch.logsumexp(neg_logits, dim=1, keepdim=True)  # [B, 1]
        
        # Loss = -log(exp(pos) / Σexp(neg)) = -pos + logsumexp(neg)
        loss = -pos_logits + neg_log_sum_exp
        
        return loss.mean()


class PolySampledSoftmaxLoss(nn.Module):
    """
    Poly Sampled Softmax Loss
    
    SampledSoftmax(InfoNCE)에 PolyLoss의 gradient boosting을 결합.
    CSAR의 양수 공간(Softplus) 제약에서도 충분한 gradient 흐름을 유지.
    
    수식:
    L = CE + ε * (1 - p_pos)
    
    여기서:
    - CE = -log(exp(s_pos) / (exp(s_pos) + Σexp(s_neg)))
    - p_pos = softmax(scores)[pos_idx]
    - ε > 0: 쉬운 샘플도 더 학습 (gradient 증폭)
    - ε < 0: 어려운 샘플에 집중 (Focal Loss 느낌)
    """
    def __init__(self, temperature=0.1, epsilon=1.0):
        """
        Args:
            temperature (float): 온도 파라미터 (낮을수록 sharp)
            epsilon (float): PolyLoss 계수 (>0이면 gradient 증폭)
        """
        super(PolySampledSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [Batch, 1] Positive 점수
            neg_scores: [Batch, K] Negative 점수
        """
        pos_logits = pos_scores / self.temperature  # [B, 1]
        neg_logits = neg_scores / self.temperature  # [B, K]
        
        # [Batch, 1 + K]: Positive가 column 0
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        # Positive 확률 (p_pos)
        probs = F.softmax(all_logits, dim=1)  # [B, 1+K]
        p_pos = probs[:, 0]  # [B]
        
        # Cross-Entropy: Positive는 항상 index 0
        labels = torch.zeros(all_logits.shape[0], dtype=torch.long, device=all_logits.device)
        ce_loss = F.cross_entropy(all_logits, labels, reduction='none')  # [B]
        
        # PolyLoss: CE + ε * (1 - p_pos)
        poly_loss = ce_loss + self.epsilon * (1 - p_pos)
        
        return poly_loss.mean()


class PolySSMLoss(nn.Module):
    def __init__(self, epsilon=1.0, tau=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau

    def forward(self, pos_scores, neg_scores):
        # 1. 온도 조절을 통한 스케일 안정화
        logits = torch.cat([pos_scores, neg_scores], dim=1) / self.tau
        
        # 2. 표준 CE (SSM의 기본 항)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        ce_loss = F.cross_entropy(logits, labels)
        
        # 3. 테일러 급수 1차 보정 항 (Poly 항)
        pt = F.softmax(logits, dim=1)[:, 0]
        poly_loss = self.epsilon * (1.0 - pt).mean()
        
        return ce_loss + poly_loss


class ScaledSampledSoftplus(nn.Module):
    """
    SSP: Softplus Sampled Softmax
    
    CSAR의 Softplus와 일관된 로스 함수.
    exp() 대신 softplus()를 사용하여 양수 공간 특성 유지.
    
    수식:
    p_pos = softplus(s_pos) / (softplus(s_pos) + Σsoftplus(s_neg))
    L = -log(p_pos)
    """
    def __init__(self, beta=1.0, eps=1e-10):
        """
        Args:
            beta: softplus의 beta 파라미터 (클수록 ReLU에 가까움)
            eps: 수치 안정성
        """
        super(ScaledSampledSoftplus, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [Batch, 1] Positive 점수
            neg_scores: [Batch, K] Negative 점수
        """
        # Softplus 변환 (exp 대신!)
        pos_sp = F.softplus(pos_scores, beta=self.beta)  # [B, 1]
        neg_sp = F.softplus(neg_scores, beta=self.beta)  # [B, K]
        
        # 양수 공간 확률
        total = pos_sp + neg_sp.sum(dim=1, keepdim=True)
        p_pos = pos_sp / (total + self.eps)
        
        # Negative Log Likelihood
        loss = -torch.log(p_pos + self.eps)
        
        return loss.mean()


class PositiveManifoldLoss(nn.Module):
    """
    TM-PML: Target Margin Positive Manifold Loss
    비율을 특정 목표(target_ratio)까지만 벌리는 로스
    - 무한 팽창 방지: 목표 도달 시 gradient 0
    - 커버리지 회복: 인기 아이템 목표 달성 → 롱테일로 학습 에너지 전이
    """
    def __init__(self, target_ratio=10.0, weight=0.1):
        super(PositiveManifoldLoss, self).__init__()
        # target_ratio: pos가 neg보다 몇 배 더 커야 하는지
        self.target_log_margin = math.log(target_ratio)
        self.w = weight

    def forward(self, pos_scores, neg_scores):
        # 1. 로그 비율 계산 (수치 안정성)
        pos_log = torch.log(pos_scores + 1e-8)
        neg_log = torch.log(neg_scores.mean(dim=1, keepdim=True) + 1e-8)
        
        current_log_ratio = pos_log - neg_log  # [B, 1]

        # 2. Hinge Loss: 목표 비율 도달하면 0
        ranking_loss = F.relu(self.target_log_margin - current_log_ratio).mean()

        # 3. Growth Loss: pos 점수가 너무 작아지는 것 방지
        growth_loss = F.relu(1.0 - pos_log).pow(2).mean()

        return ranking_loss + self.w * growth_loss


class NDCGWeightedListwiseBPR(nn.Module):
    """
    NDCG 스타일 위치 가중치가 적용된 Listwise BPR
    (명시적 네거티브 샘플링 전용)
    """
    def __init__(self, k=10, use_zscore=False):
        super().__init__()
        self.k = k
        self.use_zscore = use_zscore
    
    def forward(self, scores):
        """
        Args:
            scores (Tensor): [Batch, 1 + NumNeg] - Column 0 is Positive, Columns 1..N are Negatives
        """
        # Apply Z-score Normalization if requested
        if self.use_zscore:
            mean = scores.mean()
            std = scores.std()
            scores = (scores - mean) / (std + 1e-9)

        B = scores.size(0)
        device = scores.device
        
        # Explicit Negative Sampling: [B, 1 + NumNeg]
        # Column 0 is Positive. Columns 1..N are Negatives.
        pos_scores = scores[:, 0].unsqueeze(1) # [B, 1]
        neg_scores = scores[:, 1:]             # [B, N]
        
        # Pairwise differences
        diff = pos_scores - neg_scores         # [B, N]
        
        # Ranking: Sort all scores (Pos + Negs) row-wise
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
        ndcg_weights = 1.0 / torch.log2(neg_ranks + 2.0)
        
        # Top-K Masking
        topk_mask = neg_ranks < self.k
        final_mask = topk_mask.float()
        
        # Weighted BPR Loss
        bpr_term = -torch.log(torch.sigmoid(diff).clamp(min=1e-8))
        weighted_loss = bpr_term * ndcg_weights * final_mask
        
        return weighted_loss.sum() / final_mask.sum().clamp(min=1.0)

class ViolationAwareNDCGListwiseBPR(nn.Module):
    """
    개선된 Listwise BPR:
    - Dynamic Masking: 랭킹 위반(Violation)이 발생한 모든 Negative를 포함
    - LambdaRank Weights: |1/log(pos_rank+2) - 1/log(neg_rank+2)| 로 NDCG 영향력 반영
    """
    def __init__(self, k=10):
        super().__init__()
        self.k = k
    
    def forward(self, scores):
        B = scores.size(0)
        device = scores.device
        
        pos_scores = scores[:, 0].unsqueeze(1) # [B, 1]
        neg_scores = scores[:, 1:]             # [B, N]
        diff = pos_scores - neg_scores         # [B, N]
        
        # Ranking
        sorted_indices = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(scores)
        ranks.scatter_(
            dim=1,
            index=sorted_indices,
            src=torch.arange(scores.size(1), device=device, dtype=scores.dtype).view(1, -1).expand(B, -1)
        )
        
        pos_ranks = ranks[:, 0].unsqueeze(1)    # [B, 1]
        neg_ranks = ranks[:, 1:]                # [B, N]
        
        # NDCG 가중치 함수
        def get_ndcg_weight(r):
            return 1.0 / torch.log2(r + 2.0)
            
        # LambdaRank 가중치: Positive와 Negative를 바꿨을 때의 NDCG 차이
        w_p = get_ndcg_weight(pos_ranks)
        w_n = get_ndcg_weight(neg_ranks)
        ndcg_weights = torch.abs(w_p - w_n)
        
        # Dynamic Masking: 
        # 1. 랭킹 위반이 발생한 모든 Negative (neg_rank < pos_rank)
        # 2. 또는 랭킹 위반은 아니더라도 Top-K에 위치한 Negative
        is_violation = neg_ranks < pos_ranks
        is_topk = neg_ranks < self.k
        
        mask = (is_violation | is_topk).float()
        
        # Weighted BPR Loss
        bpr_term = -F.logsigmoid(diff)
        weighted_loss = bpr_term * ndcg_weights * mask
        
        return weighted_loss.sum() / mask.sum().clamp(min=1.0)

class TopK_NDCG_BPR(nn.Module):
    """
    Top-K 범위 내에서 NDCG 가중치를 적용한 BPR 손실
    In-Batch 샘플링용 (대각선이 Positive)
    """
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
    안정적인 In-Batch Hard BPR 손실:
    - Positive: 대각선 요소
    - Negative: 배치 내 모든 아이템
    - Hard Negative만 사용 (Positive보다 점수 높은 경우)
    - 랭킹 가중치 없음 (단순 BPR)
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores):
        """
        Args:
            scores: [B, B] (i번째 행: i번째 유저의 배치 내 아이템별 점수)
                    대각선 요소가 Positive
        """
        B = scores.size(0)
        device = scores.device
        
        # 1. Positive 점수 추출 (대각선)
        labels = torch.arange(B, device=device)
        pos = scores[labels, labels].unsqueeze(1)   # [B, 1]
        
        # 2. 모든 Negative 점수
        neg = scores                                # [B, B]

        # 3. 점수 차이 계산 (브로드캐스팅)
        diff = (pos - neg) / self.temperature       # [B, B]
        
        # 4. 자기 자신 비교 제외 마스크
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        
        # 5. Hard Negative 마스크: Positive보다 점수 높은 경우만
        hard_mask = (neg > pos).detach()
        
        mask = self_mask & hard_mask                # [B, B]
        mask = mask.float()
        
        # 6. BPR 손실 계산
        loss_map = -F.logsigmoid(diff) * mask
        
        # 7. 정규화 (유효 샘플 수로 나눔)
        return loss_map.sum() / mask.sum().clamp(min=1.0)


class DecoupledContrastiveLoss(nn.Module):
    """
    Decoupled Contrastive Learning (DCL)
    InfoNCE의 분모에서 Positive Sample을 제거하여 학습 효율 증대.
    """
    def __init__(self, temperature=0.1, weight_fn=None):
        super(DecoupledContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn # Optional: Negative에 가중치 부여 함수

    def forward(self, z1, z2):
        """
        z1, z2: [Batch, Dim] (Augmented Views)
        Cross-view Contrastive Learning
        """
        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity Matrix [B, B]
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 1. Positive Similarity (대각선)
        pos_sim = torch.diag(sim_matrix).view(B, 1) # [B, 1]
        
        # 2. Negative Similarity (Positive 제외)
        # InfoNCE와 달리 분모 합산 시 자기 자신(Positive)을 뺍니다.
        # 마스킹 방식으로 구현
        eye = torch.eye(B, device=z1.device)
        
        # exp(sim) 계산, 대각선은 0으로 처리 (합산에서 제외하기 위해)
        exp_sim = torch.exp(sim_matrix) * (1 - eye)
        
        # log(sum(exp(neg)))
        log_prob = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        
        # Loss = -pos + log(sum(neg))
        # 원래 InfoNCE: -pos + log(exp(pos) + sum(exp(neg)))
        loss = -pos_sim + log_prob
        
        return loss.mean()
        
class CosineContrastiveLoss(nn.Module):
    """
    Cosine Contrastive Loss (from UltraGCN)
    Sigmoid 없이 Cosine Similarity를 직접 최적화.
    Hard Negative Mining 효과가 내장됨.
    
    [!] 주의: 입력이 Cosine Similarity (-1~1) 범위여야 함.
        일반 추천 모델의 Dot Product 출력에는 적합하지 않음.
        Face Recognition / Image Retrieval 계열 Loss.
    """
    def __init__(self, margin=0.5, neg_weight=1.0):
        """
        Args:
            margin: Negative가 이 값보다 커지면 벌점
            neg_weight: Negative Loss 가중치
        """
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [Batch, 1] (Cosine Similarity)
            neg_scores: [Batch, K] (Cosine Similarity)
        """
        # 1. Positive Loss: (1 - pos_sim)^2
        pos_loss = (1.0 - pos_scores).pow(2).mean()
        
        # 2. Negative Loss: max(0, neg_sim - margin)^2
        neg_loss = F.relu(neg_scores - self.margin).pow(2).mean()
        
        return pos_loss + self.neg_weight * neg_loss


class DirectAULoss(nn.Module):
    """
    Direct Alignment & Uniformity Loss
    InfoNCE를 기하학적으로 해체하여 직접 최적화.
    
    - Alignment: Positive Pair 거리를 최소화 (MSE와 유사)
    - Uniformity: 모든 포인트 간의 거리를 최대화 (Gaussian Kernel 활용)
    """
    def __init__(self, gamma=2.0):
        """
        gamma: Uniformity 가중치 (보통 1.0 ~ 3.0 사이 사용)
        """
        super(DirectAULoss, self).__init__()
        self.gamma = gamma

    def forward(self, x, y):
        """
        x, y: [Batch, Dim] - 정규화된(Normalized) 임베딩이어야 함!
        """
        # 0. 임베딩 정규화 (Safety check)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        
        # 1. Alignment Loss: (x - y)^2
        # 동일한 유저/아이템의 뷰끼리는 거리가 0이 되어야 함
        align_loss = (x - y).norm(p=2, dim=1).pow(2).mean()

        # 2. Uniformity Loss: log(mean(exp(-2 * |x_i - x_j|^2)))
        # 모든 아이템이 구 표면에 고르게 퍼져야 함 (RBF 커널 스타일)
        # 메모리 효율을 위해 x와 y 각각에 대해 계산 후 평균
        uni_loss_x = self._compute_uniformity(x)
        uni_loss_y = self._compute_uniformity(y)
        uniformity_loss = (uni_loss_x + uni_loss_y) / 2
        
        return align_loss + self.gamma * uniformity_loss

    def _compute_uniformity(self, x, t=2):
        # [BUG FIX] MPS Device compatibility: torch.pdist is not implemented on MPS.
        # Manual calculation of pairwise squared distances:
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j.T
        
        # x: [B, D]
        x_norm = torch.sum(x**2, dim=1, keepdim=True) # [B, 1]
        # Pairwise distance matrix [B, B]
        # dist_matrix_sq[i, j] = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
        dist_matrix_sq = x_norm + x_norm.t() - 2 * torch.matmul(x, x.t())
        
        # torch.pdist returns only upper triangle distances (i < j).
        # We can simulate this by taking the upper triangle of the distance matrix.
        # However, for uniformity loss calculation, we can also use the full matrix (excluding diagonal).
        # Masking diagonal elements to avoid log(0) or self-distance bias if needed, 
        # but here the formula exp(-t * d^2) for d=0 is 1.
        
        # To match torch.pdist behavior exactly (returning only upper triangle):
        B = x.size(0)
        triu_indices = torch.triu_indices(B, B, offset=1, device=x.device)
        sq_pdist = dist_matrix_sq[triu_indices[0], triu_indices[1]]
        
        # Ensure non-negative due to numerical precision
        sq_pdist = torch.clamp(sq_pdist, min=0.0)
        
        return torch.log(torch.mean(torch.exp(-t * sq_pdist)) + 1e-9)

class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    - 점수(Score)가 낮을수록 Gradient를 크게 줘서 학습을 가속화 (Self-paced Weighting)
    - Margin을 통해 Positive는 더 가깝게, Negative는 더 멀게 강제함.
    
    [!] 주의: 입력이 Cosine Similarity (-1~1) 범위여야 함.
        일반 추천 모델의 Dot Product 출력에는 적합하지 않음.
        Face Recognition / Image Retrieval 계열 Loss.
    """
    def __init__(self, m=0.25, gamma=256):
        """
        Args:
            m (float): Relaxation factor (마진). 보통 0.25 사용.
            gamma (float): Scale factor (온도). 보통 64, 128, 256 등 큰 값 사용.
        """
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, pos_scores, neg_scores):
        """
        Input:
            pos_scores: [Batch, 1] (Cosine Similarity: -1 ~ 1)
            neg_scores: [Batch, K] (Cosine Similarity: -1 ~ 1)
        """
        # 1. Self-paced Weighting (가중치 계산)
        # Positive는 1에 가까워야 함 -> (1 - score)가 클수록 가중치 Up (못 맞췄으니까)
        # Negative는 0(또는 -1)에 가까워야 함 -> score가 클수록 가중치 Up (틀렸으니까)
        # detach()를 해서 가중치 자체는 학습되지 않게 고정(Gradient Stop)
        alpha_p = torch.relu(1 + self.m - pos_scores.detach()) 
        alpha_n = torch.relu(neg_scores.detach() + self.m) 

        # 2. Margin 적용
        # Positive: score - margin
        # Negative: score + margin
        delta_p = 1 - self.m
        delta_n = self.m
        
        # 3. Logit 계산 (가중치 * (점수 - 마진))
        # gamma를 곱해서 분포를 sharp하게 만듦
        p_logit = - self.gamma * alpha_p * (pos_scores - delta_p)
        n_logit = self.gamma * alpha_n * (neg_scores - delta_n)

        # 4. Loss Computation (LogSumExp Trick)
        # Loss = log(1 + sum(exp(n_logit)) * sum(exp(p_logit)))
        # 수치 안정성을 위해 logsumexp 사용
        
        # (Batch, 1) + (Batch, K) -> Broadcasting 주의 필요
        # 보통 Circle Loss는 Class-level에서 많이 쓰지만, 
        # RecSys Pairwise에서는 아래와 같이 단순화 가능:
        
        # 방식 A: Pairwise 합산 (엄밀한 수식)
        # shape: [Batch, K]
        # logic: log(1 + sum_neg(exp) * exp(pos))
        
        neg_exp = torch.exp(n_logit) # [B, K]
        pos_exp = torch.exp(p_logit) # [B, 1]
        
        # sum over negatives
        neg_term = torch.sum(neg_exp, dim=1, keepdim=True) # [B, 1]
        
        loss = self.soft_plus(torch.log(neg_term * pos_exp + 1e-8))
        
        return loss.mean()

class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks
    동일한 입력을 두 번 forward해서 나온 결과의 일관성(Consistency)을 유지.
    """
    def __init__(self, alpha=4.0):
        super(RDropLoss, self).__init__()
        self.alpha = alpha # KL-Loss의 가중치
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p_logits, q_logits):
        """
        p_logits, q_logits: 두 번의 forward pass 결과 (Logits)
        """
        # KL Divergence는 확률 분포(Softmax) 간의 거리
        p_prob = F.log_softmax(p_logits, dim=-1)
        q_prob = F.log_softmax(q_logits, dim=-1)
        
        # 양방향 KL Divergence (Symmetric)
        # KL(P||Q) + KL(Q||P)
        p_loss = self.kl_loss(p_prob, F.softmax(q_logits, dim=-1))
        q_loss = self.kl_loss(q_prob, F.softmax(p_logits, dim=-1))
        
        return self.alpha * (p_loss + q_loss) / 2

class DistributionallyRobustRenyiLoss(nn.Module):
    """
    Distributionally Robust Rényi Loss (DrRL)
    Rényi divergence로 제약된 DRO 기반. SL과 CCL을 일반화.
    
    특징:
    - Learnable β: Hard Negative Mining 자동화 (β 이하 Negative는 무시)
    - Truncated Polynomial: margin 기반 soft-thresholding
    - γ (Rényi order): 1→KL, ∞→max 연산과 유사
    - η (radius): 클수록 보수적(robust)한 학습
    """
    def __init__(self, gamma=2.0, eta=0.1, pos_target=1.0):
        """
        Args:
            gamma: Rényi order (>1). 클수록 Hard Negative에 집중
            eta: Divergence radius. 분포 불확실성의 크기
            pos_target: Positive score의 목표값 (Cosine: 1.0, Dot: 조정 필요)
        """
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.pos_target = pos_target
        
        # Derived constants
        self.gamma_star = gamma / (gamma - 1)  # conjugate exponent
        self.c_gamma = (1 + gamma * (gamma - 1) * eta) ** (1 / gamma)
        
        # Learnable threshold: 초기값을 음수로 설정하여
        # 초반에 모든 Negative가 학습에 참여하도록 함
        # 학습이 진행되면서 "쉬운" Negative를 자동으로 걸러냄
        self.beta = nn.Parameter(torch.tensor(-0.5))

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: [Batch] or [Batch, NumPos] - Positive 점수
            neg_scores: [Batch, NumNeg] - Negative 점수
        Returns:
            Scalar loss
        """
        # Ensure 2D
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)  # [B, 1]
        
        # --- Positive Loss ---
        # 목표: pos_scores를 pos_target(기본 1.0)에 가깝게
        # L2 loss: (target - score)^2
        pos_loss = ((self.pos_target - pos_scores) ** 2).mean(dim=1)  # [Batch]
        
        # --- Negative Loss (DRO with Rényi divergence) ---
        # Truncated polynomial weighting: β 이하는 무시
        # 점수가 높을수록(나쁠수록) 더 큰 penalty
        truncated = torch.relu(neg_scores - self.beta)  # [Batch, NumNeg]
        
        # 수치 안정성: 0^γ* 방지
        truncated = truncated + 1e-8
        
        # Weighted sum with polynomial
        weighted_sum = truncated.pow(self.gamma_star).mean(dim=1)  # [Batch]
        neg_term = self.c_gamma * weighted_sum.pow(1 / self.gamma_star)

        # --- Total Loss ---
        total_loss = pos_loss + neg_term
        
        return total_loss.mean()
    
    def get_beta(self):
        """현재 학습된 threshold 값 반환 (로깅용)"""
        return self.beta.item()
        
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
        단방향 InfoNCE 손실
        Args:
            view1, view2: [B, K] (증강된 뷰)
            tau: [B, 1] (적응형 온도)
        """
        B = view1.size(0)
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        
        # 코사인 유사도 / 적응형 온도 적용
        logits = (z1 @ z2.T) / tau
        
        # 라벨: 대각선이 Positive
        labels = torch.arange(B, device=z1.device)
        
        # 단방향 Cross Entropy (속도 2배, 메모리 절약)
        return F.cross_entropy(logits, labels)


class KeyQueryReconstructionLoss(nn.Module):
    """
    Key-Query Reconstruction Loss
    
    Active interest key들만 입력 embedding을 잘 재구성하도록 강제.
    Inactive key는 자연히 attention이 낮아지면서 sparse해짐.
    
    사용법:
        loss_fn = KeyQueryReconstructionLoss(lambda_rec=0.1, min_entropy=0.5)
        rec_loss, entropy = loss_fn(embeddings, interest_keys, scale)
    """
    def __init__(self, lambda_rec=0.1, min_entropy=0.5, entropy_weight=0.1):
        """
        Args:
            lambda_rec: Reconstruction loss 가중치
            min_entropy: Collapse 방지용 최소 엔트로피 (너무 sparse하면 안 됨)
            entropy_weight: Entropy penalty 가중치
        """
        super().__init__()
        self.lambda_rec = lambda_rec
        self.min_entropy = min_entropy
        self.entropy_weight = entropy_weight
    
    def forward(self, embeddings, interest_keys, scale=1.0):
        """
        Args:
            embeddings: [B, D] - 원본 임베딩 (user 또는 item)
            interest_keys: [K, D] - Interest Key 행렬
            scale: attention score 스케일 (learnable 또는 고정)
        Returns:
            total_loss: Reconstruction + Entropy Penalty
            entropy: attention weights의 entropy (로깅용)
        """
        # Compute attention: [B, K]
        attention_scores = torch.matmul(embeddings, interest_keys.t()) * scale  # [B, K]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, K]
        
        # Reconstruct embeddings: [B, D]
        reconstructed = torch.matmul(attention_weights, interest_keys)
        
        # Reconstruction loss (MSE)
        rec_loss = F.mse_loss(reconstructed, embeddings)
        
        # Entropy of attention (higher = more uniform)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        
        # Entropy penalty: 너무 sparse해지면 안 됨 (collapse 방지)
        entropy_penalty = F.relu(self.min_entropy - entropy)
        
        # Total
        total_loss = self.lambda_rec * rec_loss + self.entropy_weight * entropy_penalty
        
        return total_loss, entropy


class TopKNDGBPRLoss(nn.Module):
    """
    Top-K 기반 NDCG 스타일 위치 가중치 BPR Loss
    - 입력: pos [B,1], neg [B,N]
    - argsort 없이 Top-K만 고려
    - Top-K 밖 negative는 loss에서 제외
    - Top-K 밖 positive 페널티 부드럽게 적용
    """
    def __init__(self, k=10, pos_penalty=1.2):
        super().__init__()
        self.k = k
        self.pos_penalty = pos_penalty

    def forward(self, pos, neg):
        """
        Args:
            pos: [B,1] positive score
            neg: [B,N] negative scores
        Returns:
            scalar loss
        """
        B, N = neg.size()
        device = neg.device

        if pos.dim() == 1:
            pos = pos.unsqueeze(1)
        
        # Top-K negative
        K = min(self.k, N)
        topk_vals, topk_inds = neg.topk(K, dim=1)  # [B,K]

        # Top-K 내 위치 가중치
        positions = torch.arange(K, device=device).float()  # [0..K-1]
        ndcg_weights = 1.0 / torch.log2(positions + 2.0)  # [K]
        ndcg_weights = ndcg_weights.unsqueeze(0).expand(B, K)  # [B,K]

        # BPR Loss
        pos_exp = pos.expand_as(topk_vals)  # [B,K]
        diff = pos_exp - topk_vals
        bpr_loss = -F.logsigmoid(diff)

        # 가중치 적용
        weighted_loss = bpr_loss * ndcg_weights

        # Top-K 밖 positive 페널티 (optional)
        # pos가 negative보다 낮으면 (topk 내 최대보다 작은 경우) 부드럽게 penalty
        max_topk = topk_vals.max(dim=1, keepdim=True).values
        pos_penalty_mask = (pos < max_topk).float() * (self.pos_penalty - 1.0) + 1.0
        weighted_loss = weighted_loss * pos_penalty_mask

        return weighted_loss.mean()