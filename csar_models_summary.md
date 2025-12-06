# RecSys Framework CSAR Models 정리

이 문서는 `src/models/csar/`에 구현된 **Active CSAR (Co-Support Learning based RecSys)** 계열 모델들의 구조와 특징을 정리한 것입니다.
(Legacy 모델들은 제외되었습니다.)

---

```mermaid
graph TD
    Core[<b>Core: Co-Support Attention Layer</b><br>Embedding -> Interest Energy (Non-negative)] --> Base

    subgraph "Objective Function Strategy"
        Base{<b>Model Architecture</b>}
        Base -->|Pure Interest| Pure[<b>CSAR</b><br>Score = Interest Interaction]
        Base -->|Residual Interest| Res[<b>CSAR_R</b><br>Score = MF + Interest Interaction]
    end

    subgraph "Learning Strategy (Loss)"
        Pure -->|Pointwise| CSAR[<b>CSAR</b><br>(MSE/BCE)]
        Pure -->|Pairwise Ranking| BPR[<b>CSAR_BPR</b><br>(BPR + Ortho Loss)]
        Pure -->|Listwise / Distribution| Sampled[<b>CSAR_Sampled</b><br>(Sampled Softmax + LogQ Correction)]

        Res -->|Pointwise| CSAR_R[<b>CSAR_R</b><br>(MSE/BCE)]
        Res -->|Pairwise Ranking| RBPR[<b>CSAR_R_BPR</b><br>(BPR)]
    end
```

## 1. 핵심 레이어 (Core Layer)

### **CoSupportAttentionLayer**

- **위치**: `src/models/csar/csar_layers.py`
- **역할**: 사용자 및 아이템의 $d$-차원 임베딩을 $K$-차원의 **관심사 강도(Interest Intensity)** 벡터로 변환.
- **주요 파라미터**:
  - `num_interests` ($K$): 관심사 개수 (Latent Topics).
  - `soft_relu` (Bool): `Softplus` + `ReLU` 결합 활성화 함수 사용 여부 (Gradient Flow 개선).
  - `scale` (Bool): Attention Logit 스케일링 ($\frac{1}{\sqrt{K}}$) 적용 여부.
- **동작 원리**:
  1. 임베딩과 $K$개의 `Interaction Key` 간의 내적 계산.
  2. `Softplus` (또는 `SoftReLU`) 활성화 함수를 통해 **비음수(Non-negative) 에너지** 값으로 변환. (관심사의 '세기'는 음수가 될 수 없음)

---

## 2. 기본 모델 (Base Variants)

### **1. CSAR (Co-Support Attention RecSys)**

- **위치**: `src/models/csar/CSAR.py`
- **유형**: Pointwise Regression / Classfication
- **철학**: "유저와 아이템은 내적(Similarity)이 아니라 **공통된 관심사(Co-Support)의 총량**으로 연결된다."
- **수식**:
  $$ Score(u, i) = \sum\_{k=1}^K I_u^{(k)} \times I_i^{(k)} $$
  ($I$: Interest Intensity Vector)
- **특징**:
  - 순수하게 관심사 매칭 점수만 사용.
  - 내적 기반 모델(MF)과 달리 해석 가능성(Explainability)이 뛰어남. (어떤 관심사 $k$ 때문에 추천되었는지 확인 가능)

### **2. CSAR_R (Residual CSAR)**

- **위치**: `src/models/csar/CSAR_R.py`
- **유형**: Pointwise Regression / Classification
- **철학**: "관심사만으로는 부족하다. 기본 취향(MF) 위에 관심사(CSAR)를 더하자." (Residual Learning)
- **수식**:
  $$ Score(u, i) = \underbrace{(u \cdot i)}_{\text{MF Score}} + \underbrace{\sum I_u \times I_i}_{\text{CSAR Score}} $$
- **특징**:
  - MF의 안정적인 성능과 CSAR의 세밀한 표현력을 결합.
  - 가장 성능이 안정적인 베이스라인 모델.

### **3. CSAR_BPR (BPR Loss Variant)**

- **위치**: `src/models/csar/CSAR_BPR.py`
- **유형**: Pairwise Ranking
- **철학**: "점수의 절대값보다는 **순위(Ranking)**가 중요하다."
- **특징**:
  - 모델 구조는 `CSAR`와 동일하지만, 학습에 **BPR Loss** (또는 `DynamicMarginBPRLoss`)를 사용.
  - Implicit Feedback (클릭 여부 등) 데이터셋에서 표준 모델(`CSAR`)보다 월등한 성능.
  - 직교성 손실(`Orthogonal Loss`)을 추가하여 관심사 벡터들이 서로 다른 특징을 학습하도록 유도.

### **4. CSAR_R_BPR (Residual + BPR)**

- **위치**: `src/models/csar/CSAR_R_BPR.py`
- **유형**: Pairwise Ranking
- **철학**: `CSAR_R` 구조에 BPR Loss를 결합.
- **특징**:
  - `CSAR_R`의 강력한 표현력(Residual) + BPR의 랭킹 최적화.
  - 일반적으로 가장 **High Performance**를 보여주는 모델 조합.

---

## 3. 고급 학습 모델 (Advanced Variants)

### **5. CSAR_Sampled (Sampled Softmax)**

- **위치**: `src/models/csar/CSAR_Sampled.py`
- **유형**: Listwise / Classification (Sampled Softmax)
- **철학**: "BPR(1 vs 1)은 너무 느리고 정보가 적다. **배치 내의 모든 아이템(1 vs B)**을 비교하여 학습하자."
- **Loss**: `NormalizedSampledSoftmaxLoss` (구 `CSARLossPower`)
  - **Z-Score Normalization**: 에너지 점수 분포를 정규화하여 학습 안정성 확보.
  - **Sampled Softmax Correction**: In-Batch Negative Sampling의 편향을 보정 ($\log \frac{N}{B}$).
- **특징**:
  - Pairwise가 아닌 Listwise(유사) 방식으로 학습하여 수렴 속도가 빠름.
  - 대규모 데이터셋(Item이 많은 경우)에서 BPR보다 높은 성능을 내는 경향이 있음.
