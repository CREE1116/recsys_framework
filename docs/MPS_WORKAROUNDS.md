# MPS (Apple Silicon) Workarounds

이 문서는 프레임워크에서 Apple Silicon MPS 백엔드를 사용할 때 적용된 **우회 처리(workarounds)**를 정리합니다.

> MPS는 PyTorch 2.x부터 지원되지만 CUDA 대비 일부 연산이 미구현되어 있어, CPU fallback이나 대체 알고리즘이 필요합니다.

---

## 1. AMP (Automatic Mixed Precision)

**파일**: `src/trainer.py`

| 항목                 | CUDA                    | MPS                                          |
| :------------------- | :---------------------- | :------------------------------------------- |
| `torch.amp.autocast` | ✅ `device_type='cuda'` | ⚠️ `device_type='cpu'` (MPS autocast 미완성) |
| `GradScaler`         | ✅ 사용                 | ❌ 미지원 → `None`                           |

```python
# trainer.py:91
self.use_amp = config.get('use_amp', True) and device.type in ['cuda', 'mps']
self.scaler = GradScaler('cuda') if device.type == 'cuda' and self.use_amp else None

# trainer.py:179 — autocast 시 device_type fallback
device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
```

**영향**: MPS에서는 fp32로 실행되므로 CUDA 대비 ~2x 느릴 수 있음.

---

## 2. Sparse 연산 → CPU Fallback

**파일**: `src/models/csar/LIRALayer.py`, `src/models/general/lightgcn.py`

MPS는 **sparse tensor 연산(`addmm`, `spmm`)**을 지원하지 않습니다.

### LIRALayer: Sparse 행렬을 CPU에 보관

```python
# LIRALayer.py:296, 380, 603
if 'mps' in str(target).lower():
    self.S_sparse = S_final.cpu().to_sparse().coalesce()  # CPU 보관
```

- **추론 시**: Sparse 행렬은 CPU에, 입력 행렬은 MPS에 둔 채로 연산 → 결과만 MPS로 전송
- **대안**: 소규모 행렬은 **Dense로 변환**하여 MPS에서 직접 `torch.mm()` 수행 (EASE 수준 속도)

### LightGCN: Sparse MatMul Fallback

```python
# lightgcn.py:101-108
try:
    all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
except NotImplementedError:
    # MPS fallback: dense 변환 후 matmul
    adj_dense = self.norm_adj_matrix.to_dense()
    all_emb = torch.mm(adj_dense, all_emb)
```

- **최적화**: 노드 수 < threshold이면 아예 init에서 Dense로 변환하여 반복 변환 비용 제거

```python
# lightgcn.py:25
# 노드 수가 일정 이하이면 Dense로 변환하여 GPU 가속 극대화 (특히 MPS)
```

---

## 3. 선형대수: Cholesky / linalg.solve → CPU Offload

**파일**: `src/utils/gpu_accel.py`, `src/models/general/infinity_ae.py`

MPS는 `torch.linalg.cholesky`, `torch.linalg.solve`, `torch.linalg.lu_solve` 등을 **지원하지 않습니다**.

### EASE / Linear AE 계열: CPU Cholesky

```python
# gpu_accel.py:40
# Note: MPS doesn't support cholesky/lu_solve, so CPU-only.
```

- Gram 행렬 계산은 MPS에서 수행 (Dense matmul), Cholesky 분해만 CPU에서 수행
- 결과를 다시 MPS로 전송

### Infinity-AE: linalg.solve Fallback

```python
# infinity_ae.py:75
except:
    print("[Infinity-AE] MPS solve failed, fallback to CPU...")
    # CPU에서 solve 후 MPS로 이동
```

---

## 4. SVD: MPS Randomized SVD

**파일**: `src/utils/gpu_accel.py`

`torch.linalg.svd`는 MPS에서 동작하지만, 대규모 행렬(`n > 5000`)에서 매우 느립니다.

```python
# gpu_accel.py:282
if self.device == 'mps' and min_dim >= 5000:
    u, s, v = self._mps_randomized_svd(X_sparse, compute_k)
```

**Randomized SVD (Halko et al., 2011)**:

1. Random projection으로 차원 축소
2. Power iteration으로 정확도 향상
3. 작은 행렬에서 SVD 수행
4. **MPS GPU 위에서 Dense matmul**로 가속

---

## 5. QR 분해 → CPU Offload

**파일**: `src/models/csar/LIRALayer.py`

MPS의 QR 분해는 불안정하거나 미지원일 수 있습니다.

```python
# LIRALayer.py:510
calc_device = 'cpu' if 'mps' in str(device).lower() else device
# QR, eigendecomposition 등은 CPU에서 수행
```

---

## 6. Evaluation: gather → CPU 이동

**파일**: `src/evaluation.py`

MPS에서 `torch.gather`가 특정 텐서 조합에서 실패할 수 있습니다.

```python
# evaluation.py:441-444
# [BUG FIX] MPS Device compatibility
item_ids_cpu = item_ids.cpu()
top_indices_100_cpu = top_indices_100.cpu()
pred_lists_100 = torch.gather(item_ids_cpu, 1, top_indices_100_cpu)
```

---

## 7. Sharpening (Power Transform) → CPU 안정성

**파일**: `src/models/csar/LIRALayer.py`

고차 거듭제곱 연산이 MPS에서 수치 불안정을 일으킬 수 있습니다.

```python
# LIRALayer.py:358-359
# For stability and memory, perform sharpening on CPU if using MPS
if 'mps' in str(v.device).lower():
    v = v.cpu()
    # ... sharpening on CPU ...
    v = v.to(target_device)
```

---

## 8. GPU 가속 유틸리티 (`src/utils/gpu_accel.py`)

MPS/CUDA 환경에서 **Closed-form 모델**(EASE, LIRA, PureSVD 등)의 행렬 연산을 가속하는 유틸리티 모듈입니다. HPO에서 동일 데이터셋에 대해 반복 실험할 때 **캐시를 통해 수십 배 속도 개선**을 달성합니다.

### 8-1. `gpu_gram_solve(X_sparse, reg_lambda, rhs, device)`

**(X^T X + λI)^-1 @ rhs** 를 계산합니다. EASE 등 Linear AE 모델의 핵심 연산.

```python
from src.utils.gpu_accel import gpu_gram_solve

# EASE: P = (G + λI)^-1 G  where G = X^T X
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, rhs=None, device='auto')
```

**내부 전략** (M = 아이템 수):

| 조건       | 방법                                                | 속도          |
| :--------- | :-------------------------------------------------- | :------------ |
| M ≤ 20,000 | **Eigendecomposition 캐시**: 첫 호출 ~60s, 이후 ~1s | HPO에 최적    |
| M > 20,000 | **Cholesky per call**: ~77s/call                    | 대규모 데이터 |

### 8-2. `SVDCacheManager`

SVD 결과를 디스크에 캐싱하고, MPS에서는 Randomized SVD로 대체.

```python
from src.utils.gpu_accel import SVDCacheManager

svd_mgr = SVDCacheManager(cache_dir='data_cache', device='auto')

# k 직접 지정
u, s, v, energy = svd_mgr.get_svd(X_sparse, k=256, dataset_name='ml-1m')

# target_energy로 k 자동 결정 (예: 95% 에너지 보존)
u, s, v, energy = svd_mgr.get_svd(X_sparse, target_energy=0.95)
```

**사용 모델**: PureSVD, GF-CF, SVD-EASE, SpectralTikhonovLIRA 등

### 8-3. `_GramEigenCache` (HPO 가속)

Gram 행렬의 고유값 분해를 **메모리에 캐싱**. 같은 데이터셋에서 `reg_lambda`만 바꿔가며 HPO할 때:

- 첫 trial: 전체 Eigendecomposition (~60s)
- 이후 trials: 캐시된 고유값으로 즉시 계산 (~1s)

### 8-4. `_TruncatedSVDCache` (HPO 가속)

Truncated SVD 결과를 캐싱. LIRA 계열 모델이 `reg_lambda` HPO 시 재사용.

### 8-5. `gpu_cholesky_solve(G_np, rhs_np, device)`

CPU Cholesky (scipy)를 사용한 대칭 양정치 시스템 풀이. 블록 단위 처리로 메모리 효율적.

```
Note: MPS는 torch.linalg.cholesky를 지원하지 않으므로 CPU scipy 사용.
```

---

## 요약: MPS 우회 패턴

| 패턴                          | 적용 위치              | 방법                        |
| :---------------------------- | :--------------------- | :-------------------------- |
| **GradScaler 비활성화**       | trainer.py             | `scaler = None`             |
| **autocast device fallback**  | trainer.py             | `'cpu'` for MPS             |
| **Sparse → CPU 보관**         | LIRALayer, LightGCN    | `.cpu().to_sparse()`        |
| **Sparse mm → Dense mm**      | LightGCN               | `to_dense()` + `torch.mm()` |
| **Cholesky/solve → CPU**      | gpu_accel, infinity_ae | CPU offload                 |
| **Full SVD → Randomized SVD** | gpu_accel              | Halko algorithm on MPS      |
| **QR → CPU**                  | LIRALayer              | CPU offload                 |
| **gather → CPU**              | evaluation             | `.cpu()` before gather      |
| **Sharpening → CPU**          | LIRALayer              | 수치 안정성                 |

> **일반 원칙**: MPS에서는 **Dense 행렬곱(`torch.mm`)**이 가장 빠르고 안정적입니다. Sparse 연산, 분해(Cholesky/QR/SVD), 고급 선형대수는 CPU fallback이 현실적입니다.
