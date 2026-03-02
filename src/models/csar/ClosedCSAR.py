import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from ..base_model import BaseModel

class ClosedCSAR(BaseModel):
    """
    Closed-form CSAR
    
    LIRA의 S 행렬을 관심사 공간(M_i)과 상관관계 커널(G)로 분해:
    S ≈ M_i @ G @ M_i^T
    
    특징:
    - SVD 기반 아이템 기저 추출
    - Softplus를 통한 비음수(Non-negative) 관심사 멤버십 생성
    - OLS(최소자승법)를 통한 관심사 간 상관관계 추론
    - 학습(Backprop) 없이 분석적으로 구축됨
    """
    def __init__(self, config, data_loader):
        super(ClosedCSAR, self).__init__(config, data_loader)
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # 하이퍼파라미터
        self.num_interests = self.config['model'].get('num_interests', 64)
        self.embedding_dim = self.config['model'].get('embedding_dim', 128)
        self.reg_lambda = self.config['model'].get('reg_lambda', 500.0)
        if isinstance(self.reg_lambda, (list, np.ndarray)):
            self.reg_lambda = self.reg_lambda[0]
        self.ols_eps = self.config['model'].get('ols_eps', 1e-6)
        self.normalize = self.config['model'].get('normalize', True)

        # 결과 저장 전용 버퍼 (device 이동 고려)
        self.register_buffer('M_i', torch.zeros(self.n_items, self.num_interests))
        self.register_buffer('G', torch.zeros(self.num_interests, self.num_interests))
        
        # 훈련 데이터 CSR 행렬 (SVD용)
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # 초기 구축 (Closed-form)
        self.build()

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def build(self):
        print(f"[ClosedCSAR] Building with K={self.num_interests}, D={self.embedding_dim}, λ={self.reg_lambda} ...")
        
        # 1. SVD & LIRA S Matrix
        u, s, vt = svds(self.train_matrix_csr, k=self.embedding_dim)
        idx = np.argsort(s)[::-1]
        s, vt = s[idx], vt[idx, :]
        
        s_t = torch.from_numpy(s.astype(np.float32))
        V_k = torch.from_numpy(vt.T.astype(np.float32)) # [n_items, D]
        
        # Filter (LIRA / Ridge)
        filter_d = s_t**2 / (s_t**2 + self.reg_lambda)
        S = (V_k * filter_d) @ V_k.t() # [n_items, n_items]
        
        if self.normalize:
            d = S.abs().sum(dim=1).clamp(min=1e-8)
            d_inv = d.pow(-0.5)
            S = d_inv.unsqueeze(1) * S * d_inv.unsqueeze(0)
            
        # 2. Build M_i (Interest Membership)
        # SVD 기저의 상위 K개를 사용하여 멤버십 생성
        s_sqrt = s_t.sqrt()
        E_i = V_k * s_sqrt # Item Embedding
        
        E_top = E_i[:, :self.num_interests]
        M_i = F.softplus(E_top)
        M_i = M_i / (M_i.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. Build G (Centered OLS)
        M_c = M_i - M_i.mean(dim=0, keepdim=True)
        MtS = M_c.t() @ S @ M_c
        
        # Simple Covariance (Correlation) instead of OLS
        G = MtS
        
        # 버퍼에 할당
        self.M_i.copy_(M_i)
        self.G.copy_(G)
        
        # 재구성 에러 로깅
        S_approx = M_i @ G @ M_i.t()
        recon_err = (S - S_approx).norm() / (S.norm() + 1e-8)
        print(f"[ClosedCSAR] Done. Recon error: {recon_err:.4f}")

    def forward(self, users):
        # 유저 히스토리 가져오기
        users_np = users.cpu().numpy()
        user_history = torch.from_numpy(self.train_matrix_csr[users_np].toarray()).float().to(self.device)
        
        # interest 투영 및 점수 계산
        # score = X_u @ M_i @ G @ M_i^T
        u_interest = user_history @ self.M_i # [batch, K]
        scores = u_interest @ self.G @ self.M_i.t() # [batch, n_items]
        
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # 배치 예측 (all-item score에서 gather)
        scores = self.forward(user_ids)
        return scores.gather(1, item_ids.unsqueeze(1)).squeeze(1)

    def calc_loss(self, batch_data):
        # Closed-form 모델은 학습 루프에서의 loss가 0
        return torch.tensor(0.0, device=self.device, requires_grad=True), {}

    def get_final_item_embeddings(self):
        return self.M_i @ self.G.abs().sqrt() # 대략적인 의미 공간
