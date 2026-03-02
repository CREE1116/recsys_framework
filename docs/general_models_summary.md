# RecSys Framework General Models 정리

이 문서는 `src/models/general/`에 구현된 일반적인 추천 시스템 베이스라인 모델들의 특징과 구조를 정리한 것입니다.

---

## 1. 전통적 매트릭스 분해 (Matrix Factorization)

### **MF (Matrix Factorization)**

- **위치**: `src/models/general/mf.py`
- **특징**:
  - 가장 기본적인 Latent Factor Model.
  - 유저와 아이템을 동일한 차원의 벡터로 임베딩하고, 내적(Dot Product)을 통해 선호도를 예측.
  - $Score = u \cdot i$

### **NeuMF (Neural Matrix Factorization)**

- **위치**: `src/models/general/neumf.py`
- **특징**:
  - MF(선형 결합)와 MLP(비선형 결합)를 결합한 모델 (NCF 프레임워크).
  - GMF(Generalized Matrix Factorization) 파트와 MLP 파트가 각각의 임베딩을 갖고 최종단에서 합쳐짐.
  - 비선형적인 유저-아이템 상호작용을 포착 가능.

### **SoftplusMF**

- **위치**: `src/models/general/softplusmf.py`
- **특징**:
  - MF의 예측값에 **Softplus** 활성화 함수를 적용.
  - 점수가 항상 양수(Positive)가 되도록 강제하며, Energy-based Model의 특성을 가짐.
  - CSAR와의 공정한 비교를 위한 베이스라인으로 사용됨.

---

## 2. 그래프 신경망 (Graph Neural Networks)

### **LightGCN**

- **위치**: `src/models/general/lightgcn.py`
- **특징**:
  - NGCF(Neural Graph Collaborative Filtering)에서 불필요한 비선형 활성화와 Feature Transformation을 제거하여 경량화한 모델.
  - 이웃 노드의 임베딩을 단순히 가중 평균(Weighted Sum)하여 전파(Propagate).
  - **SOTA (State-of-the-Art)** 급의 성능을 보여주는 강력한 베이스라인.
  - Layer 수($L$)에 따라 High-order Connectivity를 학습.

---

## 3. 선형 및 오토인코더 (Linear & AutoEncoders)

### **EASE (Embarrassingly Shallow Autoencoders)**

- **위치**: `src/models/general/ease.py`
- **특징**:
  - 학습(Gradient Descent)이 없는 **Closed-form Solution** 모델.
  - 아이템 간의 가중치 행렬 $B$를 역행렬 연산으로 한 번에 계산.
  - 희소(Sparse)한 데이터셋에서 딥러닝 모델들을 압도하는 성능을 자주 보여줌.
  - Training Epoch가 필요 없음 (0 Epoch).

### **Multi-VAE (Variational Autoencoder)**

- **위치**: `src/models/general/multivae.py`
- **특징**:
  - 생성 모델(Generative Model)인 VAE를 협업 필터링에 적용.
  - 입력으로 유저의 전체 이력(Bag-of-Words)을 받고, 이를 재구성하도록 학습.
  - **Multinomial Likelihood**를 사용하여 Implicit Feedback 데이터에 강함.
  - 비선형적인 유저 선호도 분포를 모델링.

---

## 4. 프로토타입 및 앵커 기반 (Prototype & Anchor)

### **ProtoMF (Prototype-based Matrix Factorization)**

- **위치**: `src/models/general/protomf.py`
- **특징**:
  - 유저와 아이템을 직접 임베딩하는 대신, $K$개의 **프로토타입(Prototype)** 벡터들의 가중합으로 표현.
  - "이 유저는 '액션 영화' 프로토타입과 유사하다"와 같은 설명 가능성(Explainability) 제공.
  - 프로토타입 간의 직교성(Orthogonality)을 강제하여 표현의 다양성 확보.

### **ACF (Anchor-based Collaborative Filtering)**

- **위치**: `src/models/general/ACF_NLL.py`
- **변형**: `ACF_NLL` (NLL Loss), `ACF_BPR` (BPR Loss)
- **특징**:
  - ProtoMF와 유사하게 **앵커(Anchor)** 벡터를 사용하여 임베딩을 재구성.
  - **Exclusiveness Loss**: 아이템이 소수의 앵커에만 속하게 함.
  - **Inclusiveness Loss**: 모든 앵커가 골고루 사용되게 함.

---

## 5. 통계 및 기타 (Statistical & Others)

### **MostPopular**

- **위치**: `src/models/general/most_popular.py`
- **특징**:
  - 단순히 "가장 많이 소비된 아이템"을 추천.
  - 개인화가 전혀 없지만, 추천 시스템 성능 평가의 기준점(Lower Bound) 역할을 함.

### **ItemKNN**

- **위치**: `src/models/general/item_knn.py`
- **특징**:
  - 메모리 기반 협업 필터링.
  - 아이템 간의 코사인 유사도를 미리 계산해두고, 유저가 소비한 아이템과 유사한 아이템을 추천.
  - 전통적이지만 여전히 강력한 성능을 내는 베이스라인.
