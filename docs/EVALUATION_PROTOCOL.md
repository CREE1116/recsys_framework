# Evaluation Protocol

## Data Splitting

The split method is configured per dataset via `split_method` in the dataset config. Preprocessing options such as rating threshold and k-core filtering are also configurable, not hardcoded.

### Split Methods

**`loo` (Leave-One-Out):**

The last interaction per user (by timestamp; item_id breaks ties) goes to the test set, the second-to-last goes to the validation set, and the rest form the training set. Each user contributes exactly one test item, so `HitRate@K == Recall@K`.

**`temporal_ratio` (alias: `temporal`):**

Interactions are sorted chronologically per user. The last `test_ratio` fraction becomes the test set, the next `valid_ratio` fraction the validation set, and the rest training. Requires a timestamp column. Multiple test items per user are common.

**`random`:**

Same as `temporal_ratio` but split randomly without timestamp ordering. A configurable `seed` ensures reproducibility. Use when timestamp data is unavailable.

**`presplit`:**

Loads externally provided train/test files in LightGCN format (`user_id item_id item_id ...`). No k-core filtering or ID remapping is applied — IDs are assumed to be pre-processed and zero-indexed. Requires `train_file` and `test_file` config fields. No validation set is created.

### Configurable Preprocessing

The following options apply to all split methods except `presplit`:

| Option | Config Key | Default | Description |
|---|---|---|---|
| Rating threshold | `rating_threshold` | `null` (all) | Only interactions at or above this rating are kept |
| Min user interactions | `min_user_interactions` | `5` | Users with fewer interactions are excluded (k-core) |
| Min item interactions | `min_item_interactions` | `5` | Items with fewer interactions are excluded (k-core) |
| Deduplication | `dedup` | `true` | Remove duplicate (user, item) pairs |

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

The distinction matters only when a user has more than one test item (ratio-based splits):

| | `loo` | `temporal_ratio` / `random` |
|---|---|---|
| Test items per user | 1 | N (variable) |
| HitRate@K | 1 if the test item is in top-K | 1 if at least one test item is in top-K |
| Recall@K | Identical to HitRate | (# test items in top-K) / (# total test items) |
| NDCG@K | `1/log2(rank+1)` | Multi-item DCG/IDCG |
