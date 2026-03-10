# 설정 가이드

## 개요

설정은 3개의 YAML 파일로 구성되며, 아래 순서대로 병합(deep merge)됩니다. 뒤에 오는 파일이 앞의 파일을 덮어씁니다:

```
evaluation.yaml (기본값) → dataset config → model config (최우선)
```

## 파일 구조

```
configs/
├── evaluation.yaml          # 전역 평가 기본값
├── dataset/
│   ├── ml100k.yaml
│   ├── ml1m.yaml
│   └── ...
└── model/
    ├── general/             # 베이스라인 모델 (EASE, LightGCN, MF, ...)
    └── csar/                # LIRA 및 ASPIRE 계열
```

---

## 1. `evaluation.yaml` — 전역 기본값

모든 모델에 적용되는 기본 평가 설정입니다. model config에서 지정하지 않으면 자동으로 적용됩니다.

```yaml
evaluation:
  batch_size: 2048
  validation_method: "sampled"  # "full", "sampled", "uni99"
  final_method: "full"
  metrics:
    - "NDCG"
    - "Recall"
    - "HitRate"
  top_k: [5, 10, 20, 50]
  main_metric: "NDCG"
  main_metric_k: 10
  long_tail_percent: 0.8
```

---

## 2. Dataset Config — 데이터셋별 설정

### 필수 필드

| 필드 | 설명 | 예시 |
|---|---|---|
| `dataset_name` | 데이터셋 식별자 | `"ml-1m"` |
| `data_path` | 원시 데이터 파일 경로 | `"/data/ml-1m/ratings.dat"` |
| `separator` | 컬럼 구분자 | `"::"`, `"\t"`, `","` |
| `columns` | 컬럼명 | `["user_id", "item_id", "rating", "timestamp"]` |
| `min_user_interactions` | 유저당 최소 상호작용 수 (k-core) | `5` |
| `min_item_interactions` | 아이템당 최소 상호작용 수 (k-core) | `5` |
| `split_method` | 데이터 분할 방식 | 아래 참조 |
| `data_cache_path` | 캐시 저장 디렉터리 | `"./data_cache/"` |

### 선택 필드

| 필드 | 설명 | 기본값 |
|---|---|---|
| `rating_threshold` | 이 값 이상의 평점만 positive 상호작용으로 사용 | `null` (전체 사용) |
| `train_ratio` | 비율 분할 시 학습 데이터 비율 | `0.8` |
| `valid_ratio` | 비율 분할 시 검증 데이터 비율 | `0.1` |
| `has_header` | 데이터 파일 첫 줄이 헤더인지 여부 | `false` |
| `dedup` | 중복 (유저, 아이템) 상호작용 제거 | `true` |

### 분할 방식

**`loo` (Leave-One-Out):**

```yaml
split_method: "loo"
```

유저별 마지막 상호작용(타임스탬프 기준, 동점 시 item_id)을 테스트 세트로, 두 번째 마지막을 검증 세트로 사용합니다. LOO에서는 `HitRate@K == Recall@K`입니다(유저당 테스트 아이템이 1개이기 때문).

**`temporal_ratio`:**

```yaml
split_method: "temporal_ratio"
train_ratio: 0.8
valid_ratio: 0.1
```

타임스탬프 컬럼이 필요합니다. 유저별 상호작용을 시간순 정렬 후, 마지막 10%를 테스트, 다음 10%를 검증으로 사용합니다.

**`random`:**

```yaml
split_method: "random"
train_ratio: 0.8
valid_ratio: 0.1
```

`temporal_ratio`와 동일하지만 타임스탬프 없이 랜덤으로 분할합니다. 타임스탬프가 없는 데이터셋에 사용합니다.

**`presplit`:**

```yaml
split_method: "presplit"
train_file: "/data/gowalla/train.txt"
test_file: "/data/gowalla/test.txt"
```

LightGCN 형식(각 줄: `user_id item_id item_id ...`)의 외부 학습/테스트 파일을 로드합니다. k-core 필터링이나 ID 리매핑을 적용하지 않으며, ID가 이미 전처리되어 0부터 시작한다고 가정합니다. 검증 세트는 생성되지 않습니다.

### 예시: MovieLens 1M

```yaml
dataset_name: "ml-1m"
data_path: "/data/ml-1m/ratings.dat"
separator: "::"
columns: ["user_id", "item_id", "rating", "timestamp"]
rating_threshold: 4
min_user_interactions: 5
min_item_interactions: 5
split_method: "temporal_ratio"
train_ratio: 0.8
valid_ratio: 0.1
data_cache_path: "./data_cache/"
```

---

## 3. Model Config — 모델별 설정

### Closed-form 모델 (학습 루프 없음)

```yaml
model:
  name: ease
  reg_lambda: 500.0

device: "auto"
```

`train:` 블록이 없으면 Trainer가 `fit()`을 직접 호출하고 경사하강을 건너뜁니다.

### 경사하강 기반 모델

```yaml
model:
  name: lightgcn
  embedding_dim: 64
  n_layers: 3

train:
  epochs: 500
  batch_size: 1024
  lr: 0.001
  optimizer: "adam"
  loss_type: "pairwise"
  num_negatives: 1
  early_stop_patience: 40
  embedding_l2: 1.0e-4

device: "auto"
```

### 모델별 평가 설정 오버라이드

model config에서 전역 평가 설정을 덮어쓸 수 있습니다:

```yaml
evaluation:
  metrics: ["NDCG", "HitRate"]
  top_k: [10, 50]
  validation_method: "uni99"
```

---

## 데이터 캐시

전처리된 데이터는 `data_cache/`에 다음 키 형식으로 캐싱됩니다:

```
{dataset_name}_{split_method}_rt{rating_threshold}_mu{min_u}_mi{min_i}_tr{train_ratio}_vr{valid_ratio}_dedup{0|1}.pkl
```

전처리 파라미터가 변경되면 새로운 캐시 파일이 자동으로 생성됩니다. 이전 파일은 자동으로 삭제되지 않습니다.

---

## HitRate vs. Recall

| | LOO | temporal\_ratio / random |
|---|---|---|
| 유저당 정답 수 | 1개 | N개 (가변) |
| HitRate@K | 테스트 아이템이 Top-K에 있으면 1 | 테스트 아이템 중 하나라도 Top-K에 있으면 1 |
| Recall@K | HitRate와 동일 | (Top-K에 포함된 테스트 아이템 수) / (전체 테스트 아이템 수) |
| NDCG@K | `1/log2(rank+1)` | 다중 아이템 DCG/IDCG |
