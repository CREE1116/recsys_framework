# 손실 함수

모든 손실 함수는 `src/loss.py`에 구현되어 있습니다.

---

## BPRLoss

암묵적 피드백을 위한 표준 베이지안 개인화 랭킹 손실:

```
L = -log σ(s_pos - s_neg)
```

순위가 중요한 일반적인 추천 태스크에 사용합니다. 단순하고 강건합니다.

---

## DynamicMarginBPRLoss

현재 점수 차에 따라 마진이 동적으로 조정되는 BPR 변형:

```
m_dyn = α · σ(s_pos - s_neg)
L = Softplus(-(s_pos - s_neg - m_dyn))
```

모델이 이미 positive와 negative를 잘 구분하면 마진이 작아지고, 차이가 작으면 마진이 커집니다. 쉬운 샘플에 대한 기울기 신호를 강화합니다. `alpha`가 마진 스케일을 제어합니다.

---

## MSELoss

명시적 피드백(평점 예측)을 위한 Pointwise 평균 제곱 오차:

```
L = (1/N) Σ (ŷ - y)²
```

실제 평점 값이 있는 경우(예: MovieLens에서 평점 사용)에 적합합니다.

---

## SampledSoftmaxLoss (InfoNCE)

여러 negative를 동시에 처리하는 Listwise 손실:

```
L = -log [ exp(s_pos / τ) / (exp(s_pos / τ) + Σ_j exp(s_neg_j / τ)) ]
```

파라미터:
- `temperature` (τ): 분포의 선명도를 제어합니다. 낮을수록 모델이 더 확실하게 분포합니다.
- `scale_factor`: negative 점수에 곱하여 hard negative를 강조합니다 (`> 1.0`).
- `log_q_correction`: negative 샘플링의 인기도 편향을 보정하는 log-Q 보정을 적용합니다.

많은 negative 샘플을 사용하는 Top-K 최적화에 사용합니다.

---

## L2 정규화

독립적인 손실 클래스가 아닙니다. `calc_loss()` 내부에서 `self.get_l2_reg_loss(*tensors)`를 호출합니다:

```
L_reg = embedding_l2 · Σ ||x||² / (2 · B)
```

`train.embedding_l2` 설정 키로 제어됩니다. `embedding_l2 <= 0`이면 0을 반환합니다.

---

## 선택 가이드

| 상황 | 손실 함수 |
|---|---|
| 일반 추천 (암묵적 피드백) | `BPRLoss` |
| 많은 negative로 Top-K 최적화 | `SampledSoftmaxLoss` |
| Hard negative 강조 | `SampledSoftmaxLoss(scale_factor > 1)` |
| 명시적 평점 예측 | `MSELoss` |
| 잘 분리된 샘플에서 더 빠른 수렴 | `DynamicMarginBPRLoss` |
