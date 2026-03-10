# Loss Functions

All loss functions are in `src/loss.py`.

---

## BPRLoss

Standard Bayesian Personalized Ranking loss for implicit feedback:

```
L = -log σ(s_pos - s_neg)
```

Use for general recommendation tasks where rank order matters. Simple and robust.

---

## DynamicMarginBPRLoss

BPR variant with an adaptive margin that scales with the current score gap:

```
m_dyn = α · σ(s_pos - s_neg)
L = Softplus(-(s_pos - s_neg - m_dyn))
```

The margin is small when the model already separates positive from negative well, and large when the gap is close. This provides a stronger gradient signal on easy samples. `alpha` controls the margin scale.

---

## MSELoss

Pointwise mean squared error for explicit feedback (rating prediction):

```
L = (1/N) Σ (ŷ - y)²
```

Use when ground truth ratings are available (e.g., MovieLens with rating values).

---

## SampledSoftmaxLoss (InfoNCE)

Listwise loss treating multiple negatives simultaneously:

```
L = -log [ exp(s_pos / τ) / (exp(s_pos / τ) + Σ_j exp(s_neg_j / τ)) ]
```

Parameters:
- `temperature` (τ): controls the sharpness of the distribution. Lower values make the model more confident.
- `scale_factor`: multiplies negative scores, effectively up-weighting hard negatives (`> 1.0`).
- `log_q_correction`: applies log-Q correction to compensate for popularity bias in negative sampling.

Use for top-K optimization with a large negative sample budget.

---

## L2 Regularization

Not a standalone loss class. Call `self.get_l2_reg_loss(*tensors)` from inside `calc_loss()`:

```
L_reg = embedding_l2 · Σ ||x||² / (2 · B)
```

Controlled by the `train.embedding_l2` config key. Returns zero if `embedding_l2 <= 0`.

---

## Selection Guide

| Scenario | Loss |
|---|---|
| General recommendation (implicit feedback) | `BPRLoss` |
| Top-K optimization with many negatives | `SampledSoftmaxLoss` |
| Hard negative emphasis | `SampledSoftmaxLoss(scale_factor > 1)` |
| Explicit rating prediction | `MSELoss` |
| Faster convergence on well-separated samples | `DynamicMarginBPRLoss` |
