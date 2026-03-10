# 모델 레퍼런스

`src/models/`에 구현된 모델들을 설명합니다.

---

## Closed-form 모델

### EASE

**파일:** `src/models/general/ease.py`

Embarrassingly Shallow Autoencoder. 아이템-아이템 가중치 행렬 `B`를 closed-form ridge regression 해로 계산합니다:

```
B = (X^T X + λI)^{-1} X^T X,  diag(B) = 0
```

추천 점수: `X @ B`. 학습 에폭 없음. 희소 데이터셋에서 경쟁력이 있습니다. `gpu_gram_solve`로 GPU 가속 계산을 수행합니다.

### LIRA

**파일:** `src/models/csar/LIRA.py`, 레이어: `LIRALayer.py::LIRALayer`

Linear Interest covariance Ridge Analysis. 유저-유저 Gram 행렬을 통한 dual ridge regression:

```
K = X X^T,  CX = (K + λI)^{-1} X,  S = X^T CX
```

추천 점수: `X @ S`. 밀집 계산 방식이므로 소·중규모 데이터셋에 적합합니다.

### LightLIRA

**파일:** `src/models/csar/LightLIRA.py`, 레이어: `LIRALayer.py::LightLIRALayer`

SVD 기반 스펙트럼 근사 LIRA. `n_users × n_users` 행렬 역산을 피하고 저차원 SVD 부분 공간에서 연산합니다:

```
filter = σ^2 / (σ^2 + λ),  score = (X V) * filter @ V^T
```

O(nk) 추론. 대규모 데이터셋에 적합합니다. SVD 캐싱을 위해 `SVDCacheManager`를 사용합니다.

### ASPIRE

**파일:** `src/models/csar/ASPIRE.py`, 레이어: `ASPIRELayer.py::ASPIRELayer`

MNAR(Missing Not At Random) 보정을 적용한 인기도 디바이어스 아이템 유사도 모델. 노출 편향 파라미터를 추정하고 Gamma 보정을 아이템 상호작용 행렬에 적용한 후 SVD 기반 저차원 근사로 유사도를 계산합니다.

### ItemKNN

**파일:** `src/models/general/item_knn.py`

메모리 기반 협업 필터링. 코사인 아이템-아이템 유사도 행렬을 사전 계산하고, 유저의 상호작용 이력과 유사도 가중치를 결합하여 아이템을 추천합니다.

### Most Popular

**파일:** `src/models/general/most_popular.py`

비개인화 기준점 모델. 전체 아이템 상호작용 수 기준 인기 아이템을 모든 유저에게 추천합니다. 하한 기준으로 사용됩니다.

### Pure SVD

**파일:** `src/models/general/pure_svd.py`

상호작용 행렬의 절단 SVD를 통한 추천. 유저를 잠재 공간에 투영하고 투영 점수로 아이템을 랭킹합니다.

### SVD-EASE

**파일:** `src/models/general/svd_ease.py`

SVD로 근사한 EASE. 전체 Gram 행렬을 풀지 않고 저차원 SVD 부분 공간에서 스펙트럼 필터를 적용합니다. 대규모 아이템 세트에서 EASE보다 확장성이 좋습니다.

### SLIM

**파일:** `src/models/general/slim.py`

Sparse Linear Method. 좌표 하강법(coordinate descent)과 L1+L2 정규화로 희소 아이템-아이템 가중치 행렬을 학습합니다. 계산이 느릴 수 있지만 해석 가능한 희소 가중치를 생성합니다.

### GF-CF

**파일:** `src/models/general/gf_cf.py`

그래프 기반 closed-form 추천. SVD를 사용하여 상호작용 행렬에 그래프 주파수 필터를 적용합니다.

---

## 경사하강 기반 모델

### MF (Matrix Factorization)

**파일:** `src/models/general/mf.py`

BPR 또는 SampledSoftmax 손실로 학습하는 유저·아이템 임베딩 모델. 예측: `u · i^T`. 잠재 요인 모델의 기본 베이스라인입니다.

### NeuMF

**파일:** `src/models/general/neumf.py`

NCF 방식의 모델로, GMF(일반화 MF)와 MLP 경로를 결합합니다. 각 경로는 독립적인 임베딩을 가지며, 최종 예측 레이어 직전에 결합됩니다.

### LightGCN

**파일:** `src/models/general/lightgcn.py`

비선형 활성화 함수와 feature transformation 없이 그래프 합성곱만을 사용하는 경량 협업 필터링. 정규화 가중 합으로 L개 레이어에 걸쳐 이웃 임베딩을 전파합니다. 그래프 기반 추천의 강력한 베이스라인입니다.

### Multi-VAE

**파일:** `src/models/general/multivae.py`

다항 우도로 학습하는 변분 오토인코더. 유저의 전체 상호작용 이력(bag-of-words)을 입력으로 받아 확률적 잠재 표현을 통해 재구성합니다.

### ProtoMF

**파일:** `src/models/general/protomf.py`

프로토타입 기반 MF. 유저와 아이템을 K개의 공유 프로토타입 벡터의 가중 조합으로 표현합니다. 직교성 정규화를 통해 표현의 다양성을 확보합니다.

### UltraGCN

**파일:** `src/models/general/ultragcn.py`

무한 레이어 그래프 합성곱을 근사하는 제약 기반 그래프 CF. 유저-아이템 그래프에 대한 사전 계산된 제약 행렬로 임베딩 학습을 정규화합니다.

### SimGCL

**파일:** `src/models/general/simgcl.py`

그래프 대조 학습 모델. 임베딩에 균등 노이즈를 추가하여 augmented view를 생성하고 InfoNCE로 정렬합니다.

---

## 손실 함수

`BPRLoss`, `SampledSoftmaxLoss`, `MSELoss`, `DynamicMarginBPRLoss`에 대한 자세한 내용은 [loss_functions_summary.md](loss_functions_summary.md)를 참고하세요.
