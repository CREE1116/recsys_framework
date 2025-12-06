# RecSys Framework Loss Functions 정리

이 문서는 `src/loss.py` 및 각 모델 파일에 구현된 주요 손실 함수(Loss Function)들의 수식, 철학, 그리고 특징을 정리한 것입니다.

---

## 1. 랭킹 최적화 (Ranking Optimization)

### **BPR Loss (Bayesian Personalized Ranking)**

- **위치**: `src/loss.py` -> `BPRLoss`
- **수식**:
  $$ L*{BPR} = - \sum \ln \sigma( s*{pos} - s\_{neg} ) $$
- **특징**:
  - Pairwise Ranking의 표준. Positive가 Negative보다 더 높은 점수를 갖도록 최적화.
  - 가장 널리 쓰이며 안정적임.

### **Dynamic Margin BPR Loss**

- **위치**: `src/loss.py` -> `DynamicMarginBPRLoss`
- **수식**:
  $$ m*{dyn} = \alpha \cdot \sigma( s*{pos} - s*{neg} ) $$
  $$ L = \text{Softplus}( -(s*{pos} - s*{neg} - m*{dyn}) ) $$
- **특징**:
  - 학습 난이도(점수 차이)에 따라 마진($m$)을 동적으로 조절.
  - 이미 잘 맞추는 샘플은 더 확실하게 벌리고(Hard Margin), 못 맞추는 샘플은 부드럽게 학습.

---

## 2. 분류 및 대조 학습 (Classification & Contrastive)

### **InfoNCE Loss**

- **위치**: `src/loss.py` -> `InfoNCELoss`
- **수식**:
  $$ L = - \ln \frac{\exp(s*{pos}/\tau)}{\exp(s*{pos}/\tau) + \sum \exp(s\_{neg}/\tau)} $$
- **특징**:
  - 하나의 Positive와 다수의 Negative(1 vs N)를 비교.
  - 온도(Temperature, $\tau$) 파라미터를 통해 Hard Negative Mining 효과 조절 가능.

### **Normalized Sampled Softmax Loss (구 CSARLossPower)**

- **위치**: `src/loss.py` -> `NormalizedSampledSoftmaxLoss`
- **주 사용처**: `CSAR_Sampled`
- **핵심 메커니즘**:
  1. **Z-Score Normalization**: $s' = \frac{s - \mu}{\sigma}$
     - 배치 내 점수 분포를 정규화하여 학습 스케일을 고정. Gradient Explosion 방지.
  2. **Sampled Softmax Correction**: $s'_{neg} \leftarrow s'_{neg} + \ln \frac{N-1}{B-1}$
     - In-Batch Negative Sampling으로 인해 과소평가되는 분모(Partition Function)를 보정.
     - 전체 아이템에 대한 Softmax를 근사(Estimate).
- **특징**:
  - 대규모 데이터셋에서 BPR보다 빠른 수렴과 높은 성능.
  - 학습의 안정성이 매우 뛰어남 (Temperature 튜닝 불필요).

### **CSAR Loss (Log-Energy)**

- **위치**: `src/loss.py` -> `CSARLoss`
- **특징**:
  - CSAR 모델의 `Softplus` 출력(Energy)을 위해 설계됨.
  - 곱셈 형태의 Energy를 Log 변환하여 덧셈(Logit) 형태로 바꾼 뒤 Softmax 적용.

### **Entropy Adaptive InfoNCE**

- **위치**: `src/loss.py` -> `EntropyAdaptiveInfoNCE`
- **특징**:
  - 모델의 **불확실성(Entropy)**과 표현의 **품질(Orthogonality)**을 실시간으로 감지.
  - 상태에 따라 온도($\tau$)를 자동으로 조절하여 학습 스케줄링을 자동화 (Self-Paced Learning).

---

## 3. 정규화 및 보조 손실 (Regularization & Auxiliary)

### **Orthogonal Loss (직교 손실)**

- **위치**: `src/loss.py` -> `orthogonal_loss` / `src/models/csar/csar_layers.py`
- **목적**: 학습된 Latent Vector(관심사, 프로토타입 등)들이 서로 겹치지 않고 다양해지도록 강제.
- **방식**:
  - $L1$: $| \cos(v_i, v_j) |$ 최소화.
  - $L2$: $\cos^2(v_i, v_j)$ 최소화.

### **PolyLoss**

- **위치**: `src/loss.py` -> `PolyLoss`
- **특징**:
  - Cross Entropy의 다항식 확장형.
  - $\epsilon$ 파라미터를 통해 "쉬운 문제"와 "어려운 문제" 중 어디에 집중할지 조절 가능 (Focal Loss의 일반화 버전).

### **Exclusiveness & Inclusiveness Loss**

- **위치**: `src/models/general/ACF_NLL.py`
- **특징**:
  - 앵커(Anchor) 기반 모델에서 앵커 사용의 희소성(Sparsity)과 균형(Balance)을 맞추기 위한 엔트로피 기반 손실.
