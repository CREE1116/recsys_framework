# RecSys Framework Loss Functions 정리

이 문서는 현재 코드베이스(`src/loss.py`, `src/models/*.py`)에 구현된 다양한 손실 함수(Loss Function)들의 수식, 철학, 그리고 특징을 정리한 것입니다.

---

## 1. 기본 랭킹 손실 (Ranking Losses)

### **BPR Loss (Bayesian Personalized Ranking)**

- **위치**: `src/loss.py` -> `BPRLoss`
- **수식**:
  $$ L*{BPR} = - \ln \sigma( s*{pos} - s\_{neg} ) $$
  (여기서 $\sigma$는 시그모이드 함수, $s$는 점수)
- **철학**: "유저는 관측된 아이템(Positive)을 관측되지 않은 아이템(Negative)보다 더 선호할 것이다."라는 베이지안 가정을 기반으로, 두 아이템 간의 상대적 순위를 최적화합니다.
- **특징**:
  - 추천 시스템의 가장 표준적인 Pairwise Loss.
  - 0/1 등급의 암시적 피드백(Implicit Feedback) 데이터에 매우 적합함.
  - **단점**: Gradient Vanishing 문제가 발생할 수 있으며, Hard Negative(구분하기 어려운 오답)에 대한 학습 능력이 상대적으로 약함.

### **Dynamic Margin BPR Loss**

- **위치**: `src/loss.py` -> `DynamicMarginBPRLoss`
- **수식**:
  $$ m*{dyn} = \alpha \cdot \sigma( s*{pos} - s*{neg} ) $$
  $$ L*{DM-BPR} = \text{Softplus}( -(s*{pos} - s*{neg} - m\_{dyn}) ) $$
- **철학**: "이미 잘 맞추고 있는(쉬운) 샘플은 더 확실하게 벌려놓자(Margin을 키우자)." 혹은 반대로 설정에 따라 어려운 샘플에 집중하도록 유도.
- **특징**:
  - 현재 모델의 예측 차이($s_{pos} - s_{neg}$)에 따라 마진을 동적으로 조절함.
  - 학습 진행 상황에 맞춰 난이도를 조절하여 BPR의 수렴 성능을 개선.

---

## 2. 대조 학습 및 정보 이론 손실 (Contrastive & Info-Theoretic Losses)

### **InfoNCE Loss**

- **위치**: `src/loss.py` -> `InfoNCELoss`
- **수식**:
  $$ L*{InfoNCE} = - \ln \frac{\exp(s*{pos} / \tau)}{\exp(s*{pos} / \tau) + \sum*{j \in Neg} \exp(s\_{neg, j} / \tau)} $$
- **철학**: "Positive 샘플과 Negative 샘플들 사이의 상호정보량(Mutual Information) 하한을 최대화하자." (Positive는 당기고, 다수의 Negative는 밀어냄)
- **특징**:
  - BPR(1개의 Negative)과 달리 **다수의 Negative**를 한번에 고려함 (Listwise 접근에 가까움).
  - 온도 파라미터($\tau$)가 분포의 선명함(Sharpness)을 조절하여 Hard Negative Mining 효과를 냄.
  - 추천 성능(Top-K Accuracy)이 BPR보다 일반적으로 우수함.

### **CSAR Loss (Log-Energy InfoNCE)**

- **위치**: `src/loss.py` -> `CSARLoss`
- **수식**:
  $$ E(u, i) = \text{Softplus}(u) \cdot \text{Softplus}(i) $$
  $$ L\_{CSAR} = \text{CrossEntropy}( E / \tau ) $$
- **철학**: CSAR 모델은 Intensity(강도) 기반으로 점수를 매기므로, 일반적인 내적과 값이 다릅니다(0 ~ $\infty$). 이를 확률적 에너지 모델 관점에서 해석하여 학습합니다.
- **특징**:
  - **Log-Space Transform**: 곱셈 기반의 에너지를 안정적으로 학습하기 위해 Logit 관점에서 처리.
  - **Fixed Temperature**: 학습 가능한 스케일 대신 고정된 온도를 사용하여 Gradient가 임베딩 학습에만 집중되도록 강제(Gradient Stealing 방지).

### **CSAR Loss Power (Power-Energy InfoNCE)**

- **위치**: `src/loss.py` -> `CSARLossPower`
- **수식**:
  $$ s' = \frac{E(u, i) - \mu}{\sigma} \quad (\text{Global Norm}) $$
  $$ L\_{Power} = \text{CrossEntropy}( s' / \tau + \text{correction} ) $$
- **철학**: Log 변환은 큰 값들의 차이를 너무 줄여버려 랭킹 변별력을 떨어뜨릴 수 있습니다. 대신 전역 정규화(Global Normalization)와 Sampled Softmax 보정을 통해 안정적이면서도 강력한 신호를 유지합니다.
- **특징**:
  - **Global Matrix Normalization**: 배치 내 점수 행렬 자체를 정규화하여 스케일 문제를 원천 차단.
  - **Bias Correction**: 배치 내 Negative 샘플링이 전체 아이템 공간을 대표하도록 $\ln \frac{N-1}{B-1}$ 보정항 추가.

### **Entropy Adaptive InfoNCE**

- **위치**: `src/loss.py` -> `EntropyAdaptiveInfoNCE`
- **수식**:
  $$ \tau*{adapt} = \tau*{base} \cdot \exp(\alpha \cdot (\text{Uncertainty} - \text{Quality})) $$
- **철학**: "모델이 불확실해(Uncertainty)하거나 표현 품질(Orthogonality)이 낮을 때는 학습 난이도($\tau$)를 낮춰서(Soft) 안정적으로 배우고, 자신있을 때는 어렵게(Sharp) 배워라."
- **특징**:
  - **Decoupled Adaptation**: 불확실성(Entropy)과 직교성(Key Quality)이라는 두 가지 지표를 통해 온도($\tau$)를 동적으로 조절.
  - Gradient는 차단하고 상태 진단용으로만 사용하여 학습 안정성 확보.

---

## 3. 정규화 및 제약 손실 (Regularization & Constraint Losses)

### **Orthogonal Loss (직교 손실)**

- **위치**: `src/loss.py` -> `orthogonal_loss`
- **수식**:
  $$ L*{orth} = \sum*{i \neq j} | \cos(k_i, k_j) | $$
- **철학**: "서로 다른 관심사(Key Vector)들은 서로 독립적이어야 한다." (다양성 확보 및 중복 방지)
- **특징**:
  - 관심사 벡터(Key)들 간의 코사인 유사도가 0이 되도록(수직이 되도록) 강제함.
  - Disentanglement(특징 분리)를 위한 핵심 정규화 항.

### **Interest Alignment Loss (IAL)**

- **위치**: `src/models/CSAR_IAL.py`
- **수식**:
  $$ L*{align} = D*{KL}( P*{user} || Q*{item} ) = \sum P*{user} \ln \frac{P*{user}}{Q\_{item}} $$
- **철학**: "유저가 해당 포지티브 아이템을 소비했다면, 유저의 관심사 분포($P$)와 아이템이 제공하는 관심사 분포($Q$)는 유사해야 한다."
- **특징**:
  - BPR이 점수(총합)만 맞춘다면, IAL은 **내부 구성 비율(Why)**까지 맞추도록 유도함.
  - 설명 가능한 추천(Explainability)을 강화하는 효과.

### **Exclusiveness & Inclusiveness Loss**

- **위치**: `src/models/ACF_NLL.py`
- **수식**:
  - **Exc**: $H(coeffs) \rightarrow \text{min}$ (Entropy 최소화 = 한 앵커에 집중)
  - **Inc**: $H(\text{Global Mean}) \rightarrow \text{max}$ (Global Entropy 최대화 = 모든 앵커가 골고루 쓰임)
- **철학**:
  - **Exclusiveness**: 각 아이템은 명확한 소수의 특징(앵커)만 가져야 한다. (Sparse Representation)
  - **Inclusiveness**: 하지만 전체적으로 보면 모든 특징(앵커)들이 버려지지 않고 활용되어야 한다. (Collapse 방지)
- **특징**:
  - 앵커 기반 모델(ACF)에서 Representation Collapse(모든 아이템이 하나의 앵커로 쏠림)를 막는 필수적인 제약 조건.

---

## 4. 커리큘럼 및 강건성 손실 (Curriculum & Robustness Losses)

### **Curricular Consistency BPR Loss**

- **위치**: `src/models/CSAR_R_CCBPR.py`
- **수식**:
  $$ w*i = \exp(-\beta \cdot |s*{MF} - s*{CSAR}|) $$
  $$ L = \sum w_i L*{BPR} + \lambda (s*{MF} - s*{CSAR})^2 $$
- **철학**: "두 가지 뷰(MF vs CSAR)가 서로 동의하는(쉬운/확실한) 샘플부터 학습하고, 점차 어려운 샘플로 확장하자(Self-Paced Learning)."
- **특징**:
  - **Disagreement 기반 불확실성**: 두 모델의 예측 차이를 '불확실성'으로 정의.
  - **Consistency Regularization**: 두 모델이 서로 닮아가도록 강제하여 불확실성을 점진적으로 감소시킴.

### **Robust Curriculum BPR Loss (Welsch)**

- **위치**: `src/models/CSAR_R_CCBPR.py`
- **수식**:
  $$ w(r) = \exp(-\beta \cdot r^2) $$
- **철학**: 이상치(Outlier)나 노이즈 데이터에 대해 가중치를 0으로 수렴시켜 학습을 방해하지 못하게 하자(Robust Regression).
- **특징**:
  - Welsch Loss (Robust Statistics)의 가중치 함수를 차용.
  - 노이즈가 많은 데이터셋에서 일반 BPR보다 훨씬 강건함.

### **PolyLoss**

- **위치**: `src/loss.py` -> `PolyLoss`
- **수식**:
  $$ L*{Poly} = L*{CE} + \epsilon (1 - P_t) $$
- **철학**: Cross-Entropy의 테일러 급수 전개를 변형하여, 다항식(Polynomial) 형태로 Loss를 재설계. $\epsilon$ 값에 따라 쉬운 문제(Major)와 어려운 문제(Minor)에 대한 가중치를 조절.
- **특징**:
  - Focal Loss보다 더 유연하게 Gradient를 조절 가능.
  - $\epsilon > 0$이면 잘 틀리는 문제(Hard)에 집중, $\epsilon < 0$이면 확실한 문제(Easy)에 집중.

---

## 요약 (추천)

- **일반적인 추천**: `CSARLossPower` (가장 최신 기술 적용됨)
- **데이터가 Sparse할 때**: `BPRLoss` (간단하고 강력함)
- **Top-K 성능 극대화**: `InfoNCELoss` (다수의 Negative 활용)
- **노이즈가 많은 데이터**: `RobustCurriculumBPRLoss` (이상치 무시)
- **설명력이 중요할 때**: `CSAR_IAL` (관심사 정렬)
