import os
# [Windows/Mac 환경 안정성 최적화]
# Windows 환경에서 OpenMP 스레드 충돌로 인한 silent crash 방지를 위해 
# 스레드 수를 4개 정도로 보수적으로 제한합니다. (필요 시 조절 가능)
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

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
        # 우선순위: config['model']['seed'] -> config['seed'] -> 42
        self.seed = config['model'].get('seed', config.get('seed', 42))

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.device = get_device('auto')
        
        # PyTorch 생태계(추론, 평가 등)와의 완벽한 호환성을 위한 껍데기 Embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # [중요] GPU(CUDA) 사용 시 런타임 오류가 발생하는 환경이 있어 CPU로 강제합니다.
        # Windows 환경의 OpenMP 안정성을 위해 CPU 백엔드만 사용하도록 설정됨.
        self.device = torch.device('cpu')
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
        # [중요] implicit 라이브러리는 정렬된 인덱스를 기대하는 경우가 많으며, 
        # 정렬되지 않은 경우 C++ 레벨에서 Segfault가 발생할 수 있음.
        X.sum_duplicates()  # 중복 합치기
        X.sort_indices()    # 인덱스 정렬 (안정성 핵심)

        backend_name = 'CUDA GPU' if self.use_gpu else 'CPU (OpenMP)'
        print(f"[iALS] Training using {backend_name} backend.")
        start_time = time.time()
        
        # 최적화된 백엔드 연산 수행
        # Windows 환경에서 HPO 진행 시 tqdm(show_progress)이 스레드 충돌을 일으키는 경우가 있어 꺼둠.
        try:
            print(f"[iALS] Fitting engine (max_iter={self.max_iter})...")
            self.engine.fit(X, show_progress=False)
            print(f"[iALS] total training time: {time.time() - start_time:.4f}s")
        except Exception as e:
            print(f"[iALS] Error during fit: {str(e)}")
            # 만약 뻗어버리는 게 Segfault 라면 여기까지 올 수도 없겠지만, 
            # 일반적인 에러라면 여기서 잡힐 것임.
            raise e

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