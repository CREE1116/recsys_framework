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
