# GPU 가속 유틸리티

`src/utils/gpu_accel.py`는 추천 모델을 위한 디바이스 인식 선형 대수 연산을 제공합니다. 모든 함수는 사용 가능한 디바이스에 따라 CUDA, MPS, CPU로 자동 디스패치됩니다.

---

## 디바이스 디스패치

우선순위 순서:

```
CUDA → MPS (Apple Silicon) → CPU
```

`get_device(preference='auto')` 함수가 이를 구현합니다:

```python
from src.utils.gpu_accel import get_device

device = get_device('auto')   # torch.device('cuda'), 'mps', 'cpu' 중 하나 반환
device = get_device('cpu')    # CPU 강제
```

대부분의 내부 함수는 같은 로직을 따르는 `device='auto'` 인자를 받습니다.

---

## SVDCacheManager

디스크 기반 캐싱을 통해 절단 SVD를 계산하고 관리합니다. 같은 데이터셋을 사용하는 여러 모델이 SVD를 재계산하지 않고 동일한 캐시를 재사용합니다.

### 캐시 키

SVD 결과는 `data_cache/svd_{dataset}_{matrix_hash}_k{k}.pt`로 저장됩니다. 행렬 해시는 shape, nnz, 데이터 및 인덱스 배열의 샘플에 대한 MD5입니다. 데이터셋 이름이 같더라도 전처리 설정(평점 임계값, k-core 필터링 등)이 다르면 다른 캐시 파일이 생성됩니다.

### 사용법

```python
from src.utils.gpu_accel import SVDCacheManager

manager = SVDCacheManager(cache_dir='data_cache', device='auto')

# SVD 계산 또는 캐시 로드 (k개 상위 특이값/벡터)
u, s, v, total_energy = manager.get_svd(X_sparse, k=200, dataset_name='ml-1m')

# 캐시에서만 로드 (X_sparse=None 가능, 캐시가 있어야 함)
u, s, v, total_energy = manager.get_svd(dataset_name='ml-1m', k=200)
```

### `get_svd` 파라미터

| 파라미터 | 설명 |
|---|---|
| `X_sparse` | scipy sparse 행렬 (CSR). 캐시 존재 시 `None` 가능. |
| `k` | 계산할 특이값 개수. |
| `target_energy` | `k` 대신 사용: 전체 에너지의 이 비율을 포착할 만큼의 성분 수를 자동 결정. |
| `dataset_name` | 캐시 파일명에 사용. |
| `force_recompute` | `True`이면 기존 캐시를 무시하고 재계산. |

### 캐시 재사용

캐시에 `k' > k`인 결과가 있으면 절단(truncate)하여 즉시 반환합니다. 다양한 rank 값을 실험할 때 재계산을 방지합니다.

### SVD 백엔드

| 디바이스 | 조건 | 백엔드 |
|---|---|---|
| CUDA | `min(M, N) >= 5000` | 랜덤화 SVD (Halko et al.) — 네이티브 희소 CSR, `torch.sparse.mm` + `torch.linalg.qr` |
| MPS | `min(M, N) >= 5000` | 랜덤화 SVD (배치 처리) — MPS QR 불안정성 회피를 위해 고유값 분해 사용 |
| CPU / 소규모 행렬 | 항상 | `scipy.sparse.linalg.svds` (반복법) 또는 `scipy.linalg.svd` (밀집) |

CUDA 랜덤화 SVD 동작 원리:

1. scipy sparse 행렬에서 GPU CSR 텐서 구성.
2. 스케치: `Y = X @ G` (G는 랜덤 가우시안 행렬).
3. `torch.linalg.qr`로 정규직교화.
4. 파워 이터레이션으로 근사 정확도 향상.
5. 저차원 공간에 투영 후 `torch.linalg.svd`로 소규모 밀집 SVD 수행.
6. 전체 공간의 특이 벡터 복원.

### 캐시 무효화

```python
manager.invalidate()                  # 모든 SVD 캐시 파일 삭제
manager.invalidate(key='ml-1m')       # ml-1m 관련 항목만 삭제
```

---

## GramEigenCacheManager

Gram 행렬(`X^T X`)의 고유값 분해 결과를 인메모리 캐싱합니다. `gpu_gram_solve`에서 동일 행렬에 대해 다른 λ 값을 반복 적용할 때 고유값 분해 재계산을 방지합니다.

캐시 키는 행렬 구조(shape, nnz, 데이터/인덱스 샘플)의 해시로 결정되므로 데이터셋 이름이 같아도 전처리 결과가 다르면 충돌이 없습니다.

이 캐시는 Python 프로세스 메모리에만 존재하며 디스크에 저장되지 않습니다.

```python
from src.utils.gpu_accel import GramEigenCacheManager

# 보통 gpu_gram_solve를 통해 간접적으로 사용됨
# 직접 접근 (드물게 필요):
result = GramEigenCacheManager.get(X_sparse)  # (V, eigvals) 또는 None 반환
GramEigenCacheManager.put(X_sparse, V, eigvals)
GramEigenCacheManager.clear()
```

---

## gpu_gram_solve

`(X^T X + λI)^{-1} @ rhs`를 효율적으로 계산합니다. EASE, LIRA 등 closed-form 모델에서 사용됩니다.

```python
from src.utils.gpu_accel import gpu_gram_solve

# Gram 행렬의 역행렬 계산 (X^T X + λI)^{-1} 반환
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, device='auto')

# 선형 시스템 풀기: (X^T X + λI)^{-1} @ rhs
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, rhs=rhs_np, device='auto')

# GPU 텐서로 반환
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, return_tensor=True, device='auto')
```

### 디스패치 전략

| 조건 | 방법 |
|---|---|
| `M <= 15000` (첫 번째 호출) | `scipy.linalg.eigh`로 전체 고유값 분해 후 캐싱 |
| `M <= 15000` (이후 호출) | `GramEigenCache`에서 로드, 새 λ만 즉시 적용 |
| `M > 15000` | Cholesky 분해 (`gpu_cholesky_solve`) |

고유값 경로는 동일 데이터셋에 다양한 λ 값을 적용하는 HPO에서 유리합니다. 분해는 한 번만 계산하고 대각 스케일링만 변경하면 됩니다.

---

## gpu_cholesky_solve

Cholesky 분해로 `G @ X = rhs`를 풀거나 `G^{-1}`을 계산합니다. 대규모 Gram 행렬의 폴백으로 사용됩니다.

```python
from src.utils.gpu_accel import gpu_cholesky_solve

# G 역행렬 계산 (numpy 배열 반환)
G_inv = gpu_cholesky_solve(G_np, device='auto')

# G @ X = rhs 풀기
X = gpu_cholesky_solve(G_np, rhs_np=rhs, device='auto')

# GPU 텐서로 반환
X = gpu_cholesky_solve(G_np, device='auto', return_tensor=True)
```

CUDA 또는 MPS: `torch.linalg.cholesky` + `torch.cholesky_solve` 사용. OOM 발생 시 CPU로 폴백.

CPU: `scipy.linalg.cho_factor`에 `overwrite_a=True`를 사용하여 메모리를 절약합니다. 역행렬 계산 시 대규모 단위행렬 할당을 피하기 위해 블록 단위로 풀이합니다.

---

## CacheRegistry

모델은 `BaseModel.register_cache_manager()`로 캐시 매니저를 등록합니다. `Trainer`는 실행 종료 시 `CacheRegistry`를 사용하여 집계된 캐시 상태를 출력합니다:

```
[Cache] 3 entries, ~45 MB on disk
```

개별 파일 경로, 데이터셋 이름, 행렬 해시는 표시하지 않고 총 개수와 디스크 사용량만 표시합니다.

```python
# 모델 __init__에서:
self.register_cache_manager('svd', SVDCacheManager(device=self.device.type))
```

레지스트리는 등록된 각 매니저의 `summary()`를 호출하여 결과를 집계합니다.
