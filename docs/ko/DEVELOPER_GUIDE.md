# 개발자 가이드

새로운 모델을 추가하거나 학습·평가 파이프라인 및 내부 시스템을 이해하려는 연구자를 위한 가이드입니다.

---

## BaseModel 인터페이스

모든 모델은 `src/models/base_model.py::BaseModel`을 상속합니다. 구현해야 할 메서드는 다음 4가지입니다.

### `calc_loss(self, batch_data) -> (tuple, dict | None)`

경사하강 기반 모델의 매 학습 스텝에서 호출됩니다.

```python
def calc_loss(self, batch_data):
    """
    Args:
        batch_data (dict): {
            'user_id':     [B, 1],
            'pos_item_id': [B, 1],
            'neg_item_id': [B, K],
        }

    Returns:
        losses (tuple): (main_loss, reg_loss, ...)  — Trainer가 sum() 후 backward() 호출
        log_params (dict | None): 스텝별 로깅 값, 예: {'loss': value}
    """
```

손실 가중치는 모델 내부에서 곱해서 반환합니다. L2 정규화는 `self.get_l2_reg_loss(*tensors)` 유틸을 사용하세요 (config의 `train.embedding_l2` 기반).

### `forward(self, users) -> Tensor[B, n_items]`

평가용 Full-item 스코어링. 전체 아이템에 대한 점수 행렬을 반환합니다.

### `predict_for_pairs(self, user_ids, item_ids) -> Tensor[N]`

샘플링 평가(uni99)용 쌍별 스코어링. 각 (유저, 아이템) 쌍의 예측 점수를 반환합니다.

### `get_final_item_embeddings(self) -> Tensor | None`

ILD, 신규성, 다양성 메트릭 계산을 위한 최종 아이템 임베딩 행렬을 반환합니다. 해당 없으면 `None`을 반환합니다.

---

## 새 모델 추가하기

**Step 1 — 모델 파일 생성:**

```python
# src/models/general/my_model.py
from ..base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.dim = config['model'].get('embedding_dim', 64)
        # self.device는 BaseModel에서 자동 설정

    def calc_loss(self, batch_data): ...
    def forward(self, users): ...
    def predict_for_pairs(self, user_ids, item_ids): ...
    def get_final_item_embeddings(self): ...
```

**Step 2 — 모델 레지스트리 등록:**

```python
# src/models/__init__.py
from .general.my_model import MyModel

MODEL_REGISTRY = {
    ...
    'my_model': MyModel,
}
```

**Step 3 — 설정 파일 생성:**

```yaml
# configs/model/general/my_model.yaml
model:
  name: "my_model"
  embedding_dim: 64

train:
  epochs: 200
  batch_size: 1024
  lr: 0.001
  optimizer: "adam"
  loss_type: "pairwise"
  num_negatives: 1
  early_stop_patience: 40
  embedding_l2: 1.0e-5

device: "auto"
```

**Step 4 — 실행:**

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml1m.yaml \
  --model_config configs/model/general/my_model.yaml
```

---

## Closed-form (비경사하강) 모델

해석적으로 파라미터를 계산하는 모델은 `calc_loss()` 대신 `fit()`을 구현합니다. Trainer는 `fit()`이 정의되어 있으면 평가 전에 한 번 호출하며, config에 `train:` 블록이 없으면 경사하강 루프를 건너뜁니다.

```python
class MyClosedFormModel(BaseModel):
    def fit(self, data_loader):
        # 아이템 유사도 행렬, SVD 등 계산
        ...

    def calc_loss(self, batch_data):
        # Trainer가 이 메서드의 존재를 요구함
        return (torch.tensor(0.0, device=self.device),), None
```

**Trainer 실행 흐름:**

```
trainer.run()
  ├── model.fit() 있으면 → 한 번 호출
  ├── config에 'train:' 있으면 → _train_loop() (경사하강)
  └── evaluate()
```

---

## 모델에서 GPU 가속 사용하기

텐서 할당에는 `self.device`(BaseModel에서 설정)를 사용합니다. SVD나 Gram 행렬 연산에는 `src/utils/gpu_accel.py` 유틸리티를 사용합니다:

```python
from src.utils.gpu_accel import SVDCacheManager, gpu_gram_solve

# __init__에서:
manager = SVDCacheManager(device=self.device.type)
self.register_cache_manager('svd', manager)

# fit() 또는 __init__에서:
u, s, v, total_energy = manager.get_svd(X_sparse, k=200, dataset_name=dataset_name)
```

자세한 내용은 [GPU_ACCEL.md](GPU_ACCEL.md)를 참고하세요.

---

## Trainer 흐름

```
Trainer.run()
  ├── model.fit()         (정의된 경우)
  ├── _train_loop()       (train config 있는 경우)
  │   ├── calc_loss() per batch
  │   ├── optimizer.step()
  │   └── 매 N 에폭 evaluate() (early stopping)
  └── final evaluate()
```

메트릭, 손실, 모델 체크포인트는 `trained_model/{dataset}/{model}/`에 자동으로 저장됩니다.

---

## 손실 함수

`src/loss.py`에 구현된 손실 함수:

| 클래스 | 유형 | 설명 |
|---|---|---|
| `BPRLoss` | Pairwise | 표준 BPR: `-log σ(s_pos - s_neg)` |
| `DynamicMarginBPRLoss` | Pairwise | 점수 차에 따라 마진이 조정되는 BPR |
| `MSELoss` | Pointwise | 명시적 피드백을 위한 평균 제곱 오차 |
| `SampledSoftmaxLoss` | Listwise | 온도·log-Q 보정 지원 InfoNCE |

모델 내 사용 예:

```python
from src.loss import BPRLoss

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.loss_fn = BPRLoss()

    def calc_loss(self, batch_data):
        pos_scores = ...
        neg_scores = ...
        loss = self.loss_fn(pos_scores, neg_scores)
        reg = self.get_l2_reg_loss(self.user_emb.weight, self.item_emb.weight)
        return (loss, reg), {'bpr': loss.item()}
```

---

## 평가

모델별 또는 전역으로 설정 가능한 평가 프로토콜 3가지:

| 프로토콜 | 설명 |
|---|---|
| `full` | 전체 아이템 스코어링 후 학습 이력 마스킹, Top-K 추출 |
| `sampled` | 랜덤 유저 부분집합에 대해 전체 아이템 스코어링 |
| `uni99` | 유저당 1개 positive + 99개 랜덤 negative |

메트릭: `NDCG`, `HitRate`, `Recall`, `Precision`, `Coverage`, `ILD`, `Novelty`, `GiniIndex`, `LongTailCoverage`, `LongTailRatio`.

---

## 데이터 파이프라인

```
data_loader.py
  ├── _load_or_process_data()
  │   ├── 캐시 히트 → _load_from_cache()
  │   └── 캐시 미스 → _process_data() → _save_to_cache()
  ├── get_train_loader()
  ├── get_validation_loader()
  └── get_interaction_graph()
```

데이터 캐시 키 형식: `{dataset}_{split_method}_rt{threshold}_mu{min_u}_mi{min_i}_tr{train_ratio}_vr{valid_ratio}_dedup{0|1}.pkl`

전처리 파라미터가 변경되면 새로운 캐시 파일이 자동으로 생성됩니다.

---

## 일괄 HPO

여러 모델·데이터셋에 대해 베이지안 HPO를 일괄 실행:

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/results
```

탐색 설정 구조:

```yaml
datasets:
  - "configs/dataset/ml1m.yaml"

seeds: [42, 43, 44]

searches:
  - name: "EASE"
    model_config: "configs/model/general/ease.yaml"
    params:
      - name: "model.reg_lambda"
        type: "float"
        range: "0.1 100000"
        log: true
    n_trials: 60
    metric: "NDCG@20"
```

파라미터 타입: `float`, `int`, `categorical`.

출력: `output/{dataset}/{model}/best_params.json`, `best_val_metrics.json`, 시드별 모델 디렉터리.
