# Evaluation Protocol

## Data Splitting

| Setting | Value | Description |
|---|---|---|
| Split method | Leave-One-Out (LOO) | Last interaction per user → test set |
| Tie-breaker | item_id | Deterministic split when timestamps are equal |
| Rating threshold | ≥ 4 | Only interactions rated 4 or above are treated as positive |
| Min user interactions | 5 | Users with fewer interactions are excluded |
| Min item interactions | 5 | Items with fewer interactions are excluded |

---

## Negative Sampling

| Setting | Value | Description |
|---|---|---|
| Exclusion set | Train history only | Test items may appear as negatives (RecBole standard) |
| Sampling strategy | Independent uniform | No shared candidate pool within a batch |

---

## Evaluation Methods

| Method | Description |
|---|---|
| `full` | Score all items; mask train interactions; extract top-K |
| `sampled` | Score all items on a random subset of users |
| `uni99` | 1 positive + 99 uniform random negatives per user |

Train and validation interactions are masked from the ranking at test time. The test item itself is not masked.

---

## Metrics

**Accuracy:**
- `NDCG@K` — Normalized Discounted Cumulative Gain
- `HitRate@K` — 1 if the test item appears in top-K
- `Recall@K` — Fraction of test items in top-K (equal to HitRate for LOO)
- `Precision@K`
- K values: 5, 10, 20, 50

**Diversity and fairness:**
- `Coverage@K` — Fraction of all items that appear in at least one recommendation list
- `ILD@K` — Intra-List Diversity (mean pairwise distance within recommendation lists, based on item embeddings)
- `GiniIndex@K` — Gini coefficient of item recommendation frequency
- `LongTailCoverage@K` — Coverage restricted to the bottom 20% popularity items
- `LongTailRatio@K` — Fraction of recommendations that are long-tail items

**Novelty:**
- `Novelty@K` — Self-information-based novelty (higher for less popular items)

---

## Default Hyperparameters

| Parameter | Value |
|---|---|
| `batch_size` | 1024 |
| `early_stop_patience` | 40 |
| `embedding_l2` | 1e-4 |
| `optimizer` | AdamW |
| `lr` | 0.001 |
| `main_metric` | NDCG@10 |
| `max_epochs` | 500 |
| `device` | auto (CUDA > MPS > CPU) |

---

## HitRate vs. Recall

Under LOO splitting (one test item per user), HitRate and Recall are identical. Under ratio-based splitting (multiple test items per user), they differ:

- **HitRate@K**: 1 if at least one test item is in top-K
- **Recall@K**: (# test items in top-K) / (# total test items)
