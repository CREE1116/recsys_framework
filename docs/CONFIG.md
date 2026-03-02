# Configuration Guide

## 개요

설정은 3개의 YAML 파일로 구성되며, **아래 순서대로 병합(deep merge)** 됩니다.
뒤에 오는 설정이 앞의 설정을 덮어씁니다.

```
evaluation.yaml (기본값) → dataset config (데이터셋 특화) → model config (최종 결정)
```

## 파일 구조

```
configs/
├── evaluation.yaml          # 마스터 평가 설정 (모든 모델 공통)
├── dataset/
│   ├── ml100k.yaml          # 데이터셋별 설정
│   ├── ml1m.yaml
│   └── ...
└── model/
    ├── general/             # 일반 모델 (EASE, MF, LightGCN 등)
    └── csar/                # CSAR/LIRA 계열 모델
```

---

## 1. `evaluation.yaml` — 마스터 평가 설정

모든 모델에 적용되는 기본 평가 설정입니다. 개별 model config에서 지정하지 않아도 자동 적용됩니다.

```yaml
evaluation:
  batch_size: 2048
  validation_method: "sampled" # "full", "sampled", "uni99"
  final_method: "full" # "full", "uni99"
  metrics:
    - "NDCG"
    - "Recall"
    - "HitRate"
    # ...
  top_k: [5, 10, 20, 50]
  main_metric: "NDCG"
  main_metric_k: 10
  long_tail_percent: 0.8
```

---

## 2. Dataset Config — 데이터셋별 설정

### 필수 필드

| 필드                    | 설명                             | 예시                                            |
| ----------------------- | -------------------------------- | ----------------------------------------------- |
| `dataset_name`          | 데이터셋 식별자                  | `"ml-100k"`                                     |
| `data_path`             | 데이터 파일 절대경로             | `"/path/to/u.data"`                             |
| `separator`             | 구분자                           | `"\t"`, `"::"`, `","`                           |
| `columns`               | 컬럼명                           | `["user_id", "item_id", "rating", "timestamp"]` |
| `min_user_interactions` | 유저 최소 상호작용 수 (k-core)   | `5`                                             |
| `min_item_interactions` | 아이템 최소 상호작용 수 (k-core) | `5`                                             |
| `split_method`          | 데이터 분할 방식                 | 아래 참조                                       |
| `data_cache_path`       | 캐시 저장 경로                   | `"./data_cache/"`                               |

### 선택 필드

| 필드               | 설명                              | 기본값            |
| ------------------ | --------------------------------- | ----------------- |
| `rating_threshold` | 이 값 이상만 상호작용으로 간주    | `null` (미필터링) |
| `train_ratio`      | 학습 데이터 비율 (ratio split 시) | `0.8`             |
| `valid_ratio`      | 검증 데이터 비율 (ratio split 시) | `0.1`             |
| `has_header`       | 데이터 파일 첫 줄이 헤더인지      | `false`           |

### Split Method별 필요 설정

#### `loo` (Leave-One-Out)

```yaml
split_method: "loo"
# train_ratio, valid_ratio 불필요
# timestamp 있으면 시간순, 없으면 item_id순 정렬
```

- **평가 특성**: ground_truth = 유저당 1개
- **HR = Recall** (정답이 1개이므로 동일)

#### `temporal_ratio` (시간순 비율 분할)

```yaml
split_method: "temporal_ratio"
train_ratio: 0.8
valid_ratio: 0.1
# timestamp 컬럼 필수
```

- **평가 특성**: ground_truth = 유저당 N개
- **HR ≠ Recall** (HitRate: 하나라도 맞으면 1, Recall: 맞춘 비율)

#### `random` (랜덤 비율 분할)

```yaml
split_method: "random"
train_ratio: 0.8
valid_ratio: 0.1
# timestamp 없는 데이터셋용 (e.g., lastfm_2k)
```

- **평가 특성**: `temporal_ratio`와 동일

### 예시: MovieLens 100K

```yaml
dataset_name: "ml-100k"
data_path: "/path/to/ml100k/u.data"
separator: '\t'
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

모델 하이퍼파라미터와 학습 설정을 포함합니다.

### 비학습 모델 (EASE, LIRA, ItemKNN 등)

```yaml
model:
  name: lira # MODEL_REGISTRY 키
  reg_lambda: [500.0] # 리스트: grid search 대상

# train 섹션 없음 → 직접 fit() 후 evaluate()
device: "auto"
```

### 학습 모델 (MF, LightGCN, CSAR 등)

```yaml
model:
  name: mf
  embedding_dim: 64

train:
  epochs: 500
  batch_size: 1024
  lr: 0.001
  optimizer: "adam"
  loss_type: "pairwise" # "pairwise" or "pointwise"
  num_negatives: 1
  early_stop_patience: 40

device: "auto"
```

### evaluation Override (선택)

model config에서 evaluation을 지정하면 마스터 설정을 덮어씁니다:

```yaml
# 이 모델만 특별한 메트릭으로 평가
evaluation:
  metrics: ["NDCG", "HitRate"]
  top_k: [10, 50]
```

---

## 캐시 시스템

데이터 전처리 결과는 `data_cache/`에 캐싱됩니다.

**캐시 키 형식**: `{dataset_name}_{device}_{split_method}_rt{rating_threshold}_mu{min_u}_mi{min_i}_tr{train_ratio}_vr{valid_ratio}.pkl`

> ⚠️ `split_method`, `rating_threshold`, `train_ratio` 등을 변경한 경우 기존 캐시가 자동으로 무효화되어 새로 생성됩니다.

---

## HR vs Recall 차이

|                   | LOO                     | temporal_ratio / random           |
| ----------------- | ----------------------- | --------------------------------- |
| ground_truth 크기 | 1                       | N (가변)                          |
| HitRate@K         | 정답이 top-K에 있으면 1 | 정답 중 하나라도 top-K에 있으면 1 |
| Recall@K          | = HitRate (동일)        | (맞춘 개수) / (전체 정답 수)      |
| NDCG@K            | 1/log2(rank+1)          | multi-item DCG / IDCG             |
