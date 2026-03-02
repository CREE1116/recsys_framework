# RecSys Framework 개발자 가이드

새로운 모델을 추가하거나, 프레임워크의 내부 구조를 이해하려는 연구자를 위한 가이드입니다.

---

## 1. 모델 구현 (BaseModel Interface)

모든 모델은 `src/models/base_model.py`의 `BaseModel`을 상속받고, **4개 필수 메서드**를 구현합니다.

### 1-1. `calc_loss(self, batch_data)` — 학습

```python
def calc_loss(self, batch_data):
    """
    Args:
        batch_data (dict): {'user_id': [B,1], 'pos_item_id': [B,1], 'neg_item_id': [B,K]}

    Returns:
        (losses, params_to_log):
            - losses (tuple): (main_loss, reg_loss, ...)  ← Trainer가 sum() 후 backward()
            - params_to_log (dict): {'loss': value, ...}  ← 자동 그래프 생성
    """
```

- **튜플 규약**: Trainer는 `sum(losses)` → `backward()`. 가중치는 모델에서 곱해서 반환.
- **L2 규제**: `self.get_l2_reg_loss(*tensors)` 유틸 사용 (embedding_l2 config 기반)

### 1-2. `forward(self, users)` — Full Ranking 평가

- 입력: 유저 ID 텐서
- 출력: `[batch_size, n_items]` 점수 행렬

### 1-3. `predict_for_pairs(self, user_ids, item_ids)` — Uni99 평가

- 입력: `(user_id, item_id)` 쌍
- 출력: 해당 쌍의 예측 점수 `[N]`

### 1-4. `get_final_item_embeddings(self)` — 임베딩 분석

- 학습 후 최종 아이템 임베딩 반환 (시각화, ILD 계산 등에 사용)

---

## 2. 새로운 모델 추가하기

### Step 1: 모델 파일 생성

```python
# src/models/general/my_model.py
from ..base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        # 하이퍼파라미터: config['model'] dict에서 가져오기
        self.dim = config['model'].get('embedding_dim', 64)
        ...

    def calc_loss(self, batch_data): ...
    def forward(self, users): ...
    def predict_for_pairs(self, user_ids, item_ids): ...
    def get_final_item_embeddings(self): ...
```

### Step 2: 레지스트리 등록

```python
# src/models/__init__.py
from .general.my_model import MyModel

MODEL_REGISTRY = {
    ...
    'my_model': MyModel,
}
```

### Step 3: 설정 파일 생성

```yaml
# configs/model/general/my_model.yaml
model:
  name: "my_model"
  embedding_dim: 64

train: # 없으면 비학습 모델로 인식
  batch_size: 1024
  epochs: 100
  lr: 0.001
  loss_type: "pairwise"
  embedding_l2: 1.0e-5
```

### Step 4: 실행

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml100k.yaml \
  --model_config configs/model/general/my_model.yaml
```

---

## 3. 비학습 모델 (Closed-form)

`fit()` 메서드를 구현하면 Trainer가 자동으로 호출합니다. `train:` 블록 없으면 바로 평가.

```python
class MyClosedFormModel(BaseModel):
    def fit(self, data_loader):
        """DataLoader를 받아 모델 파라미터를 직접 계산"""
        # EASE: (X^T X + λI)^-1 X^T X
        # ItemKNN: 코사인 유사도 행렬
        ...
```

**Trainer 실행 흐름:**

```
trainer.run()
  ├── fit() 있으면 호출        ← EASE, ItemKNN, UltraGCN 등
  ├── train 설정 있으면 _train_loop()  ← SGD 모델
  └── 없으면 바로 evaluate()   ← 비학습 모델
```

---

## 4. 손실 함수

`src/loss.py`에 4개 핵심 Loss가 정의되어 있습니다:

| Loss                   | 용도                                        |
| :--------------------- | :------------------------------------------ |
| `BPRLoss`              | 표준 Pairwise ranking                       |
| `DynamicMarginBPRLoss` | 동적 마진 BPR                               |
| `MSELoss`              | Pointwise (명시적 피드백)                   |
| `SampledSoftmaxLoss`   | InfoNCE (temperature, logQ correction 지원) |

사용법: 모델의 `__init__`에서 인스턴스화, `calc_loss`에서 호출.

---

## 5. 평가 시스템

### 3가지 프로토콜

| 방식      | 설명                             | Config                       |
| :-------- | :------------------------------- | :--------------------------- |
| `full`    | 전체 아이템 스코어링 후 Top-K    | `validation_method: full`    |
| `uni99`   | 1 positive + 99 random negatives | `validation_method: uni99`   |
| `sampled` | 전체 유저의 일부만 평가          | `validation_method: sampled` |

### 메트릭

NDCG, HitRate, Recall, Precision, Coverage, ILD, Novelty, GiniIndex, PopRatio@K, LongTailCoverage, LongTailRatio

설정은 `configs/evaluation.yaml`에서 통합 관리.

---

## 6. 데이터 파이프라인

```
data_processing.py (순수 함수)        data_loader.py (오케스트레이터)
├── load_raw_data()                   ├── __init__: 캐시 로드 or 전처리
├── filter_interactions() (k-core)    ├── _process_data(): 순수 함수 호출
├── dedup_interactions()              ├── _save_to_cache() / _load_from_cache()
├── remap_ids()                       ├── get_train_loader()
├── split_leave_one_out()             ├── get_validation_loader()
├── split_temporal_ratio()            ├── get_final_loader()
├── split_random()                    └── get_interaction_graph()
└── build_history_dicts()
```

- **캐시 키**: `{dataset}_{split}_{threshold}_{min_u}_{min_i}_{ratio}_{dedup}.pkl`
- **네거티브 샘플링**: collate_fn에서 배치 단위 생성 (set-based O(N) 필터링)

---

## 7. 일괄 HPO (Batch Bayesian Optimization)

하나의 YAML 파일로 **여러 모델 × 여러 데이터셋 × 멀티시드** HPO를 한 번에 실행합니다.

### 실행

```bash
cd scripts/
uv run python run_all_smart_searches.py \
  --config ../configs/paper_baselines_search.yaml \
  --output_dir ../output/paper_baselines
```

### Search Config 구조 (`paper_baselines_search.yaml`)

```yaml
datasets:
  - "configs/dataset/ml1m.yaml"
  - "configs/dataset/ml20m.yaml"

seeds: [42, 43, 44, 45, 46] # 멀티시드 평균으로 안정적 평가

summary_metrics: # 데이터셋별 요약 테이블에 표시할 메트릭
  - "NDCG@20"
  - "Recall@20"
  - "Coverage@20"

searches:
  - name: "EASE"
    method: "bayesian"
    model_config: "configs/model/general/ease.yaml"
    params:
      - name: "model.reg_lambda"
        type: "float"
        range: "0.1 100000"
        log: true # log scale 탐색
    n_trials: 60 # Optuna trial 수
    patience: 20 # early stopping patience
    metric: "NDCG@20" # 최적화 대상 메트릭

  - name: "LightGCN"
    method: "bayesian"
    model_config: "configs/model/general/lightgcn.yaml"
    params:
      - name: "model.n_layers"
        type: "int"
        range: "1 3"
      - name: "train.embedding_l2"
        type: "categorical"
        range: "0.01 0.005 0.001 0.0005 0.0001"
      - name: "model.embedding_dim"
        type: "categorical"
        range: "32 48 64 96 128"
    n_trials: 60
    patience: 20
    metric: "NDCG@20"
```

### 파라미터 타입

| type          | 설명                         | range 형식         |
| :------------ | :--------------------------- | :----------------- |
| `float`       | 연속값                       | `"min max"`        |
| `int`         | 정수                         | `"min max"`        |
| `categorical` | 이산 선택지                  | `"val1 val2 val3"` |
| `int_min_dim` | 데이터셋 크기 기반 자동 범위 | `log: true`        |

### 출력 구조

```
output/paper_baselines/
├── {dataset_name}/
│   ├── {model_name}/
│   │   ├── best_params.json       # 최적 하이퍼파라미터
│   │   ├── best_val_metrics.json  # 검증 메트릭
│   │   └── BEST_{model}_seed_{N}/ # 시드별 최종 모델
│   └── dataset_summary_{dataset}.json  # 데이터셋별 모델 비교표
```

### 워크플로우

1. 각 search 항목에 대해 Optuna 베이지안 최적화 실행
2. 최적 하이퍼파라미터 발견 후 **모든 시드로 재실행**
3. 시드별 결과를 평균하여 최종 메트릭 산출
4. 데이터셋별 요약 JSON 자동 생성 (모델 간 비교)
