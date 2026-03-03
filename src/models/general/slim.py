import torch
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm
from ..base_model import BaseModel
import time


class _SLIMMatrixCache:
    """HPO trial 간 X_csc 재사용 캐시 (데이터셋이 같으면 X도 동일)."""
    _cache: dict = {}

    @classmethod
    def _key(cls, X_csr):
        d = X_csr.data
        idx = X_csr.indices
        ptr = X_csr.indptr
        return hash((X_csr.shape, X_csr.nnz,
                     d[:10].tobytes() if len(d) >= 10 else d.tobytes(),
                     idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                     ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))

    @classmethod
    def get(cls, X_csr):
        return cls._cache.get(cls._key(X_csr))

    @classmethod
    def put(cls, X_csr, X_csc):
        cls._cache[cls._key(X_csr)] = X_csc
        print(f"[SLIM-Cache] X_csc cached ({X_csc.shape})")

    @classmethod
    def clear(cls):
        cls._cache.clear()



def _solve_column(j, X_csc, X_csr, y_all, alpha, l1_ratio, positive, max_iter, tol):
    """
    sklearn ElasticNet으로 item j에 대한 column을 풀고 w_j를 반환.
    표준 SLIM coordinate descent solver.
    X_csr: CSR 형식 (ElasticNet 입력)
    X_csc: CSC 형식 (column zeroing에 효율적)
    y_all: X를 dense로 변환한 전체 matrix (n_users, n_items) — None이면 X_csc에서 추출
    """
    from sklearn.linear_model import ElasticNet
    import warnings

    y = X_csc[:, j].toarray().ravel().astype(np.float64)

    # 자기 자신(item j)을 feature에서 제거: CSC에서 column zeroing은 효율적
    X_mod = X_csc.copy()
    X_mod[:, j] = 0
    X_mod = X_mod.tocsr()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=positive,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol,
            copy_X=False,
            warm_start=False,
        )
        model.fit(X_mod, y)

    coef = model.coef_.astype(np.float32)
    coef[j] = 0.0
    return coef


class SLIM(BaseModel):
    """
    SLIM (Sparse Linear Methods for Top-N Recommender Systems)
    표준 구현: column-wise sklearn ElasticNet (coordinate descent) + joblib 병렬화.
    Reference: Ning & Karypis, ICDM 2011

    Optimization per column j:
      min_{w} ||X[:,j] - X @ w||^2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2
      s.t. w_j = 0, w >= 0
    """

    def __init__(self, config, data_loader):
        super(SLIM, self).__init__(config, data_loader)

        self.alpha     = config['model'].get('alpha', 0.1)
        self.l1_ratio  = config['model'].get('l1_ratio', 0.5)
        self.positive  = config['model'].get('positive_constraint', True)
        self.max_iter  = config['model'].get('max_iter', 1000)
        self.tol       = float(config['model'].get('tol', 1e-4))
        self.n_jobs    = config['model'].get('n_jobs', -1)

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.W = None
        self.register_buffer('W_tensor', torch.empty(0, 0))
        self.train_matrix_csr = None

        self._log(f"Initialized: alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
              f"positive={self.positive}, max_iter={self.max_iter}, n_jobs={self.n_jobs}")

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        X = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        self._log(f"Interaction matrix: {X.shape}, nnz={X.nnz:,}")
        return X

    def fit(self, data_loader):
        print(f"\n{'='*60}")
        self._log("Starting standard ElasticNet solver (coordinate descent)...")
        print(f"{'='*60}")
        start_time = time.time()

        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = self.train_matrix_csr
        M = self.n_items

        self._log(f"Solving {M} columns in parallel (n_jobs={self.n_jobs})...")
        # CSC 포맷은 column zeroing에 효율적 — 캐시 확인 후 1회만 변환
        X_csc = _SLIMMatrixCache.get(X)
        if X_csc is None:
            print(f"[SLIM] Converting X to CSC format (first time for this dataset)...")
            X_csc = X.tocsc()
            _SLIMMatrixCache.put(X, X_csc)
        else:
            print(f"[SLIM] Using cached CSC matrix.")

        results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(_solve_column)(
                j, X_csc, None, None, self.alpha, self.l1_ratio, self.positive,
                self.max_iter, self.tol
            )
            for j in tqdm(range(M), desc="[SLIM] Columns", unit="col")
        )

        W_out = np.stack(results, axis=1).astype(np.float32)  # (M, M)
        self.W = W_out
        self.register_buffer('W_tensor', torch.from_numpy(W_out).to(self.device))

        elapsed = time.time() - start_time
        nnz = np.sum(W_out > 1e-8)
        sparsity = 1.0 - nnz / (M * M)
        print(f"\n{'='*60}")
        self._log("Training complete!")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - W nnz: {nnz:,} / {M*M:,} (sparsity={sparsity:.4f})")
        print(f"{'='*60}\n")

    def forward(self, user_ids, item_ids=None):
        if self.W is None:
            raise RuntimeError("[SLIM] Model not fitted. Call fit() first.")

        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)

        # GPU inference: W_tensor를 self.device에 올려두고 user input도 같이 올림
        if self.W_tensor.numel() == 0:
             self.register_buffer('W_tensor', torch.from_numpy(self.W).to(self.device))

        user_input = torch.from_numpy(
            self.train_matrix_csr[u_ids_np].toarray().astype(np.float32)
        ).to(self.device)

        with torch.no_grad():
            scores = user_input @ self.W_tensor  # GPU matmul

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            else:
                item_ids = item_ids.to(self.device)
            batch_idx = torch.arange(len(u_ids_np), device=self.device)
            return scores[batch_idx, item_ids]

        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, requires_grad=True),), None

    def get_train_matrix(self, data_loader):
        if self.train_matrix_csr is None:
            self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        return self.train_matrix_csr

    def get_final_item_embeddings(self):
        # SLIM의 W는 item-item weight matrix로 ILD 계산용 embedding에 부적합
        return None