import os
# [Mac 환경 최적화] 라이브러리 로드 전 OpenMP 스레드 수를 강제 할당하여 
# 싱글 코어로 도는 현상 방지 (사용 중인 Mac의 성능에 따라 8~10 정도로 조절 가능)
# os.environ['OPENBLAS_NUM_THREADS'] = '8'
# os.environ['OMP_NUM_THREADS'] = '8'

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import implicit
from implicit.gpu.als import AlternatingLeastSquares as GPU_ALS
from implicit.cpu.als import AlternatingLeastSquares as CPU_ALS
from ..base_model import BaseModel
from ...utils.gpu_accel import get_device
import time

class iALS(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.embedding_dim = config['model'].get('embedding_dim', 128)
        self.reg_lambda = config['model'].get('reg_lambda', 0.01)
        self.alpha = config['model'].get('alpha', 40.0)
        self.max_iter = config['model'].get('max_iter', 15)
        self.seed = config['model'].get('seed', 42)

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.device = get_device('auto')

        # PyTorch 생태계(추론, 평가 등)와의 완벽한 호환성을 위한 껍데기 Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        # Device에 따른 엔진 라우팅 
        # (MPS는 C++ backend에서 지원하지 않으므로 무조건 CPU로 빠지게 설정)
        if self.device.type == 'cuda':
            self.use_gpu = True
            ALS_Engine = GPU_ALS
        else:
            self.use_gpu = False
            ALS_Engine = CPU_ALS

        # [핵심 수정] calculate_training_loss=False 로 변경 완료.
        # 이 옵션이 꺼져 있어야 매 이터레이션마다 극심한 연산 오버헤드가 발생하지 않음.
        self.engine = ALS_Engine(
            factors=self.embedding_dim,
            regularization=self.reg_lambda,
            alpha=self.alpha,
            iterations=self.max_iter,
            calculate_training_loss=False,  
            random_state=self.seed
        )

    def fit(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        
        # implicit 라이브러리 입력 형식에 맞춘 가중치 (기본 상호작용은 1.0)
        values = np.ones(len(train_df), dtype=np.float32)

        # (n_users x n_items) 크기의 CSR Matrix 생성
        X = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

        backend_name = 'CUDA GPU' if self.use_gpu else 'CPU (OpenMP)'
        print(f"[iALS] Training started using {backend_name} backend...")
        start_time = time.time()
        
        # 최적화된 백엔드 연산 수행 (show_progress=True로 설정하면 내부적으로 tqdm 바 생성됨)
        self.engine.fit(X, show_progress=True)
        
        print(f"[iALS] total training time: {time.time() - start_time:.4f}s")

        # --- 훈련 완료 후 임베딩 동기화 ---
        u_factors = self.engine.user_factors
        i_factors = self.engine.item_factors

        # CUDA 환경일 경우 CuPy 배열로 나올 수 있으므로 안전하게 Numpy로 내리기
        if hasattr(u_factors, 'get'):
            u_factors = u_factors.get()
        if hasattr(i_factors, 'get'):
            i_factors = i_factors.get()

        # PyTorch Parameter로 복사
        with torch.no_grad():
            self.user_embedding.weight.copy_(torch.from_numpy(u_factors))
            self.item_embedding.weight.copy_(torch.from_numpy(i_factors))
            
        self.user_embedding = self.user_embedding.to(self.device)
        self.item_embedding = self.item_embedding.to(self.device)

    def forward(self, user_ids, item_ids=None):
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids, device=self.device)

        users = self.user_embedding(user_ids)

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            items = self.item_embedding(item_ids)
            return (users * items).sum(dim=-1)

        return users @ self.item_embedding.weight.t()

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.data