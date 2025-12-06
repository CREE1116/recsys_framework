# RecSys Framework CSAR Models 정리

이 문서는 `src/models/`에 구현된 모든 **CSAR (Co-Support Learning based RecSys)** 계열 모델들의 구조, 철학, 그리고 특징을 정리한 것입니다.

---

## 1. 기본 모델 (Base Models)

### **CSAR (Co-Support Attention RecSys)**

- **위치**: `CSAR.py`
- **철학**: "유저와 아이템은 내적(Dot Product)이 아니라, **공통된 관심사(Co-Support)의 세기(Intensity)**로 연결되어야 한다."
- **구조**:
  - `CoSupportAttentionLayer`: 임베딩을 $K$개의 관심사 벡터(Non-negative Intensity)로 변환.
  - Score = $\sum (\text{UserInterest} \times \text{ItemInterest})$
- **특징**:
  - ReLU/Softplus 활성화를 사용하여 '음수 없는' 관심사 강도를 측정.
  - 내적 기반 MF와 달리, '왜(Why)' 추천되었는지 설명 가능(Explainability).

### **CSAR_R (Residual CSAR)**

- **위치**: `CSAR_R.py`, `CSAR_R_BPR.py`
- **철학**: "관심사만으로 모든 것을 설명할 순 없다. 기본 취향(MF) 위에 구체적 관심사(CSAR)를 얹자." (Residual Connection)
- **구조**:
  - Score = $\text{MF}(u, i) + \text{CSAR}(u, i)$
  - MF: 전반적인 선호도 (Global Bias)
  - CSAR: 구체적인 관심사 매칭 (Local Detail)
- **특징**:
  - 가장 안정적이고 성능이 좋은 베이스라인.
  - MF가 뼈대를 잡고 CSAR이 디테일을 채우는 상호보완적 구조.

---

## 2. 구조적 변형 (Structural Variants)

### **CSAR_Deep (Hierarchical CSAR)**

- **위치**: `CSAR_Deep.py`
- **철학**: "관심사는 계층적이다. (예: 영화 -> 액션 -> 마블 영화)"
- **구조**:
  - 다중 `CoSupportAttentionLayer`를 적층(Stacking).
  - 예: $d \to K_1 \to K_2$ 순으로 차원을 변환하며 더 추상적인 상위 관심사를 학습.
- **특징**:
  - 단순 $K$개 관심사가 아닌, 깊이 있는 관심사 계층 구조를 학습 가능.

### **CSAR_V / CSAR_VR (Variational CSAR)**

- **위치**: `CSAR_V.py`, `CSAR_VR.py` (Residual 버전)
- **철학**: "데이터가 부족한(Sparse) 유저는 전역적인 평균(Global Prior)을 참고해야 한다." (Bayesian Additive Smoothing)
- **구조**:
  - `VariationalInterestLayer` 사용.
  - **Personal Interest** (개인 관측값) + **Global Prior** (전역 사전지식) 구조.
  - $I_{hybrid} = \text{Softplus}(I_{personal}) + \text{Softplus}(I_{global})$
- **특징**:
  - Sparse 유저는 Personal 값이 작아 Global Prior(ACF 효과)가 주도함.
  - Dense 유저는 Personal 값이 커서 Global Prior가 무시됨(CSAR 효과).
  - VAE의 가우시안 샘플링 방식이 아니라, 에너지 기반의 베이지안 스무딩 방식을 사용함.

### **CSAR_R_Softmax (Probabilistic CSAR)**

- **위치**: `CSAR_R_Softmax.py`
- **철학**: "관심사는 강도(Intensity)가 아니라 확률(Probability)이다."
- **구조**:
  - Softplus 대신 Softmax를 사용하여 관심사 가중치의 합을 1로 제한.
  - Cross Entropy Loss와 결합하여 분류(Classification) 문제처럼 접근.
- **특징**:
  - 관심사의 총량이 제한되어 있어, 모든 분야에 관심을 가질 수 없음 (선택과 집중).

---

## 3. 정규화 및 희소성 모델 (Regularization & Sparsity)

### **CSAR_R_L0 (L0 Regularized)**

- **위치**: `CSAR_R_L0.py`
- **철학**: "유저는 수많은 관심사 중 **극히 일부(Sparse)**에만 반응한다. 나머지는 0이어야 한다."
- **특징**:
  - **L0 Gating**: Hard-Concrete 분포를 사용하여 유저별로 필요 없는 관심사 스위치를 아예 꺼버림(0).
  - 결과적으로 유저마다 "활성화된 관심사 개수"가 다르게 학습됨.
  - 노이즈 제거 효과가 탁월함.

### **CSAR_STE (Straight-Through Estimator)**

- **위치**: `CSAR_STE.py`
- **철학**: "휴리스틱 없이, Top-K 선택 과정 자체를 미분 가능하게 만들자."
- **구조**:
  - Forward: 엄격하게 Top-K개의 관심사만 선택 (Masking).
  - Backward: SoftmaxGradient를 통해 선택되지 않은 관심사로도 그라디언트 전파.
- **특징**:
  - L0 Gating보다 더 직접적으로 "상위 K개" 제약을 강제함.

### **CSAR_R_Lasso (L1 Regularized)**

- **위치**: `CSAR_R_Lasso.py`
- **철학**: "공짜 점심은 없다. CSAR(상세 관심사)를 쓰려면 대가(Penalty)를 치러라."
- **특징**:
  - 관심사 활성화 값의 합(L1 Norm)을 손실 함수에 추가.
  - L0보다 구현이 간단하고 학습이 안정적이지만, 완전히 0이 되지는 않는 경향이 있음(Soft Sparsity).

### **CSAR_R_UBR (Uncertainty-Based Regularization)**

- **위치**: `CSAR_R_UBR.py`
- **철학**: "MF와 CSAR이 서로 동의하지 않는다면, 그 예측은 불확실한 것이다."
- **특징**:
  - **Disagreement**: $|Score_{MF} - Score_{CSAR}|$를 불확실성 지표로 사용.
  - 불확실한 샘플(동의하지 않는 샘플)은 학습 가중치를 낮춤(Loss Attenuation).
  - 이상치(Outlier)에 매우 강함.

### **CSAR_R_Confidence (Confidence Gating)**

- **위치**: `CSAR_R_Confidence.py`
- **철학**: "자신 없으면 CSAR 쓰지 말고, 기본(MF)만 써라."
- **구조**:
  - 유저의 관심사 엔트로피(Entropy)를 측정.
  - 엔트로피가 낮으면(확실하면) CSAR 점수를 반영하고, 높으면(산만하면) 무시.
  - $Score = Score_{MF} + \alpha \cdot Score_{CSAR}$ ($\alpha$는 신뢰도)
- **특징**:
  - CSAR이 오히려 노이즈가 되는 상황(Cold User 등)을 방지.

---

## 4. 학습 전략 변형 (Learning Strategy Variants)

### **CSAR_R_KD (Knowledge Distillation)**

- **위치**: `CSAR_R_KD.py`
- **철학**: "안정적인 선배(MF)가 흔들리는 후배(CSAR)를 가르친다."
- **특징**:
  - MF 파트(Teacher)의 예측 분포를 CSAR 파트(Student)가 따라가도록 추가적인 Loss 부여.
  - 학습 초기 CSAR의 불안정한 수렴을 MF가 가이드하여 성능 부스팅.

### **CSAR_R_contrastive**

- **위치**: `CSAR_R_contrastive.py`
- **철학**: "같은 유저(아이템)의 관심사는 조금 변형되어도 본질이 유지되어야 한다."
- **특징**:
  - **Adaptive Contrastive Loss**: 불확실성에 따라 온도($\tau$)를 조절하며 대조 학습 수행.
  - 데이터가 극도로 희소(Sparse)한 상황에서 임베딩 품질을 높이는 데 효과적.

---

## 5. 손실 함수 특화 (Loss Specific)

### **CSAR_Sampled (InfoNCE Specialized)**

- **위치**: `CSAR_Sampled.py`
- **철학**: "BPR(순위)보다 InfoNCE(분류)가 학습 신호가 풍부하다."
- **특징**:
  - `CSARLossPower`를 기본 Loss로 채택.
  - 배치 내 Negative 샘플링을 효율적으로 수행하며, Softmax의 분모를 근사.
  - 대규모 데이터셋 학습에 유리.

### **CSAR_BPR_CE / CSAR_R_BPR**

- **위치**: `CSAR_BPR_CE.py`, `CSAR_R_BPR_SoftRelu.py` 등
- **철학**: 모델 구조보다는 **손실 함수의 최적 조합**을 탐구.
- **특징**:
  - **BPR_CE**: Ranking(BPR)과 Classification(InfoNCE/CE)를 동시에 수행하여 Multi-task 효과를 노림.
  - **BPR_SoftRelu**: Softplus(Smooth ReLU) 등을 활성화 함수로 사용하여 미분 가능성과 학습 안정성을 높인 버전.

---

## 요약 (추천)

- **가장 무난한 시작**: `CSAR_R` (또는 `CSAR_R_BPR`)
- **설명력이 중요할 때**: `CSAR` (Residual 없이 순수 관심사만 사용)
- **노이즈가 많은 데이터**: `CSAR_R_L0` (불필요한 관심사 가지치기) 또는 `CSAR_R_UBR` (이상치 무시)
- **데이터가 너무 적을 때**: `CSAR_R_contrastive` (대조 학습으로 표현력 강화)
