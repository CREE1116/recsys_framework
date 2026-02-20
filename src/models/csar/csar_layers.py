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
            

class CSAR_basic(nn.Module):
    """
    CSAR Basic Layer
    - scale은 여기서만 관리 (중복 제거)
    - get_membership: softplus 기반 NMF-style membership
    """
    def __init__(self, num_interests, embedding_dim, reg_lambda=500.0, normalize=True):
        super(CSAR_basic, self).__init__()
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        self.reg_lambda = reg_lambda
        self.normalize = normalize

        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(1.0))  # 여기서만 scale 관리
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)

    def get_membership(self, embs):
        """softplus membership → NMF 성질 (non-negative, part-based decomposition)"""
        logits = torch.einsum('...d,kd->...k', embs, self.interest_keys) * self.scale
        return F.softplus(logits)

    @property
    def orthogonality(self):
        """
        Interest key들의 직교성 지표. 1.0 = 완전 직교.
        off-diagonal cosine similarity 평균 기반.
        """
        K = F.normalize(self.interest_keys, p=2, dim=-1)
        sim = torch.matmul(K, K.t())
        n = sim.size(0)
        off_diag = sim - torch.eye(n, device=sim.device)
        return 1.0 - off_diag.abs().sum() / (n * n - n)

    def forward(self, embedding_tensor):
        return self.get_membership(embedding_tensor)

    def get_gram_matrix(self, reg_lambda=None,
                        use_ridge=False, M_u=None, M_i=None, X=None):
        """
        Inter-interest propagation matrix.

        [Mode 1] use_ridge=False (default):
            G = D^{-1/2} (K @ K.T) D^{-1/2}  (symmetric degree normalization)

        [Mode 2] use_ridge=True (OLS closed-form):
            G* = (M_u.T M_u)^{-1} M_u.T X M_i (M_i.T M_i + λI)^{-1}
        """
        if use_ridge:
            assert M_u is not None and M_i is not None and X is not None, \
                "use_ridge=True requires M_u, M_i, X"

            lam = reg_lambda if reg_lambda is not None else self.reg_lambda

            if X.is_sparse:
                X = X.to_dense()

            MuT_Mu = M_u.t() @ M_u
            MiT_Mi = M_i.t() @ M_i

            lhs = torch.linalg.solve(MuT_Mu, M_u.t() @ X @ M_i)
            rhs = torch.linalg.solve(
                MiT_Mi + lam * torch.eye(self.num_interests, device=M_i.device),
                lhs.t()
            ).t()

            return rhs, None

        # Mode 1: K @ K.T + symmetric degree normalization
        K = self.interest_keys
        G_raw = torch.matmul(K, K.t())

        d = G_raw.abs().sum(dim=1)
        d_inv_sqrt = torch.where(d > 1e-12, torch.rsqrt(d), torch.ones_like(d))

        if self.normalize:
            G = d_inv_sqrt.view(-1, 1) * G_raw * d_inv_sqrt.view(1, -1)
        else:
            G = G_raw

        return G, d_inv_sqrt

class CSAR_CLLayer(nn.Module):
    """
    CSAR Contrastive Learning Layer
    Aligns prototype correlations (K@K.T) with data-driven correlations (G_tilde).
    """
    def __init__(self, num_interests, embedding_dim, ema_alpha=0.9):
        super(CSAR_CLLayer, self).__init__()
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        self.ema_alpha = ema_alpha
        
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        self.membership_scale = nn.Parameter(torch.tensor(math.sqrt(embedding_dim)))
        
        # G matrix buffers (Moved from Model class)
        self.register_buffer("G_ema", torch.eye(num_interests))
        self.register_buffer("G_tilde", torch.eye(num_interests))
        self.register_buffer("G_norm", torch.eye(num_interests))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.interest_keys)
        
    def update_G(self, X, E_u, E_i):
        """OLS G update (via lstsq) + EMA + Centering + Normalization"""
        with torch.no_grad():
            # 1. Get raw memberships
            M_u = self.get_membership(E_u)
            M_i = self.get_membership(E_i)
            
            # 2. OLS Closed-form G* = (MuT Mu)^-1 MuT X Mi (MiT Mi)^-1
            # Note: lstsq fallback to CPU for stability and MPS compatibility.
            A_cpu = (M_u.t() @ M_u).cpu()           # [K, K]
            B_cpu = (M_u.t() @ X @ M_i).cpu()       # [K, K]
            C_cpu = (M_i.t() @ M_i).cpu()           # [K, K]

            # G* = A^-1 B C^-1
            G_left_cpu = torch.linalg.lstsq(A_cpu, B_cpu).solution
            G_star_cpu = torch.linalg.lstsq(C_cpu.t(), G_left_cpu.t()).solution.t()
            
            G_star = G_star_cpu.to(self.interest_keys.device)
            
            # 3. EMA: G_ema = alpha * G_ema + (1-alpha) * G_star
            self.G_ema.copy_(self.ema_alpha * self.G_ema + (1.0 - self.ema_alpha) * G_star)
            
            # 4. Centering: G_tilde = G_ema - mean(G_ema) (Used for teacher signal in alignment)
            self.G_tilde.copy_(self.G_ema - self.G_ema.mean())
            
            # 5. Symmetric Normalization for G_ema
            d = self.G_ema.abs().sum(dim=1)
            d_inv_sqrt = torch.where(d > 1e-12, torch.rsqrt(d), torch.zeros_like(d))
            self.G_norm.copy_(d_inv_sqrt.view(-1, 1) * self.G_ema * d_inv_sqrt.view(1, -1))
        
    def get_membership(self, embs):
        """m = softplus(scale * E @ K^T)"""
        logits = torch.matmul(embs, self.interest_keys.t()) * self.membership_scale
        return F.softplus(logits)
        
    def get_alignment_loss(self, G_teacher, tau_student=0.1, tau_teacher=0.1):
        """
        L_G = InfoNCE(G_teacher, K @ K.T)
        Encourages prototype similarities to follow semantic correlations via contrastive alignment.
        """
        K = F.normalize(self.interest_keys, p=2, dim=-1) # [K, d]
        S_student = torch.matmul(K, K.t()) # [K, K]
        
        # Softmax targets from teacher (G_teacher) - Detached to ensure directional distillation
        targets = F.softmax(G_teacher / tau_teacher, dim=-1).detach()
        
        # Log-softmax outputs from student (S_student)
        outputs = F.log_softmax(S_student / tau_student, dim=-1)
        
        # InfoNCE style cross-entropy loss: - sum( p * log(q) )
        loss = -(targets * outputs).sum(dim=-1).mean()
        return loss

class CSAR_DINOLayer(CSAR_CLLayer):
    """
    CSAR DINO Layer: Self-distillation on G matrix.
    Student: Current differentiable OLS G*
    Teacher: Centered & Sharpened EMA G_ema
    """
    def __init__(self, num_interests, embedding_dim, ema_alpha=0.9):
        super(CSAR_DINOLayer, self).__init__(num_interests, embedding_dim, ema_alpha)

    def get_differentiable_G(self, X, E_u, E_i):
        """
        Calculates G* (OLS) in a differentiable way.
        Note: We force CPU for linalg.solve to avoid MPS backward pass issues.
        """
        M_u = self.get_membership(E_u)
        M_i = self.get_membership(E_i)
        
        # Move to CPU for differentiable solve (stable fallback)
        device = M_u.device
        M_u_cpu = M_u.cpu()
        M_i_cpu = M_i.cpu()
        # self.X is already dense and usually small enough to stay on CPU/Device
        # But we ensure it's on CPU for this calculation
        X_cpu = X.cpu()
        
        MuT_Mu = M_u_cpu.t() @ M_u_cpu
        B = M_u_cpu.t() @ X_cpu @ M_i_cpu
        G_left = torch.linalg.solve(MuT_Mu + 1e-6 * torch.eye(self.num_interests), B)
        
        MiT_Mi = M_i_cpu.t() @ M_i_cpu
        G_star = torch.linalg.solve(MiT_Mi + 1e-6 * torch.eye(self.num_interests), G_left.t()).t()
        
        return G_star.to(device)

    @staticmethod
    def koleo_loss(mem: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        KoLeo Spread Loss (DINOv2 implementation based)
        mem: (N, K) or (N, 1, K) - batch membership vectors
        Returns: scalar loss
        """
        if mem.dim() == 3:
            mem = mem.squeeze(1)
            
        # 1. L2 normalization (project to unit sphere)
        mem_norm = F.normalize(mem, p=2, dim=-1, eps=eps)  # (N, K)

        # 2. Nearest neighbor search based on dot product (approximate distance)
        # mem_norm @ mem_norm.t() gives cosine similarity (which is 1 - 0.5 * distance^2 for unit vectors)
        dots = mem_norm @ mem_norm.t()                        # (N, N)
        n = mem_norm.size(0)
        # Mask diagonal (self-similarity) to -inf or -1 to find *other* nearest neighbor
        dots.fill_diagonal_(-1.0)
        
        nn_idx = dots.max(dim=1).indices                      # (N,) nearest neighbor index

        # 3. Calculate actual L2 distances to nearest neighbors
        # d^2 = ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y = 1 + 1 - 2*dot = 2(1 - dot)
        # But we use explicit pdist for clarity or just dot-to-distance formula
        nn_dots = dots[torch.arange(n), nn_idx]
        distances = torch.sqrt(torch.clamp(2.0 * (1.0 - nn_dots), min=eps)) # (N,)
        
        loss = -torch.log(distances + eps).mean()
        return loss

    def get_dino_loss(self, X, E_u, E_i, tau_student=0.1, tau_teacher=0.1):
        """
        DINO-style loss using KL-Divergence: KL(Teacher || Student)
        """
        # 1. Student G* (Differentiable)
        G_student = self.get_differentiable_G(X, E_u, E_i)
        
        # 2. Teacher target (Detached, Centered, Sharpened)
        G_teacher = self.G_ema.detach()
        # Simple Centering: Just use current G_ema mean
        G_teacher_centered = G_teacher - G_teacher.mean(dim=0, keepdim=True)
        
        # Target distribution (Teacher)
        targets = F.softmax(G_teacher_centered / tau_teacher, dim=-1)
        
        # Log-Softmax distribution (Student)
        outputs_log = F.log_softmax(G_student / tau_student, dim=-1)
        
        # 3. KL-Divergence: KL(Teacher || Student)
        # kl_div(input, target) expects log_softmax for input and softmax for target
        loss = F.kl_div(outputs_log, targets, reduction='batchmean')
            
        return loss
        