# Loss Functions 정리

현재 `src/loss.py`에 구현된 손실 함수들의 수식과 용도입니다.

---

## 1. BPR Loss (Bayesian Personalized Ranking)

- **클래스**: `BPRLoss`
- **수식**: $L_{BPR} = -\ln \sigma(s_{pos} - s_{neg})$
- **용도**: 암시적 피드백(Implicit Feedback) 데이터에서의 표준 Pairwise ranking loss
- **특징**: 가장 널리 쓰이는 추천 손실 함수. 단순하고 강건함

---

## 2. Dynamic Margin BPR Loss

- **클래스**: `DynamicMarginBPRLoss`
- **수식**:
  - $m_{dyn} = \alpha \cdot \sigma(s_{pos} - s_{neg})$
  - $L = \text{Softplus}(-(s_{pos} - s_{neg} - m_{dyn}))$
- **용도**: 쉬운 샘플에는 큰 마진을 요구하여 BPR 수렴 성능 개선
- **파라미터**: `alpha` (마진 스케일링)

---

## 3. MSE Loss

- **클래스**: `MSELoss`
- **수식**: $L = \frac{1}{N} \sum (y_{pred} - y_{true})^2$
- **용도**: 명시적 피드백(Explicit Feedback, Rating Prediction) 데이터
- **사용**: Pointwise loss_type 설정 시

---

## 4. Sampled Softmax Loss (InfoNCE)

- **클래스**: `SampledSoftmaxLoss`
- **수식**: $L = -\ln \frac{\exp(s_{pos}/\tau)}{\exp(s_{pos}/\tau) + \sum_{j} \exp(s_{neg,j}/\tau)}$
- **용도**: 다수의 Negative를 한 번에 고려하는 Listwise 접근. Top-K 성능 극대화
- **파라미터**:
  - `temperature` ($\tau$): 분포 선명도 (낮을수록 sharp)
  - `scale_factor`: Negative 가중치 (>1이면 Hard Negative 강화)
  - `log_q_correction`: 인기편향 보정 (Popularity-based sampling bias correction)

---

## 5. 모델 내부 정규화

### L2 Regularization

- **위치**: `BaseModel.get_l2_reg_loss(*tensors)`
- **수식**: $L_{l2} = \lambda \cdot \frac{\sum ||x||^2_2}{2 \cdot B}$
- **설정**: `train.embedding_l2` YAML 키
- **용도**: 임베딩 파라미터의 과적합 방지. 각 모델의 `calc_loss`에서 직접 호출

### Orthogonal Loss

- **위치**: 각 CSAR 모델 내부
- **수식**: $L_{orth} = \sum_{i \neq j} |\cos(k_i, k_j)|$
- **용도**: 관심사 벡터(Interest Key) 간 직교성 강제 (다양성 확보)

---

## 선택 가이드

| 상황               | 추천 Loss                            |
| :----------------- | :----------------------------------- |
| 일반적인 추천      | `BPRLoss`                            |
| Top-K 성능 극대화  | `SampledSoftmaxLoss`                 |
| Hard Negative 강조 | `SampledSoftmaxLoss(scale_factor>1)` |
| 명시적 피드백      | `MSELoss`                            |
| 수렴 속도 개선     | `DynamicMarginBPRLoss`               |
