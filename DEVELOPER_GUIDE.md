# RecSys Framework 개발자 가이드

이 문서는 프레임워크의 내부 구조를 이해하고, 새로운 모델이나 기능을 추가하려는 연구자/개발자를 위한 상세 가이드입니다.

---

## 1. 모델 구현 (BaseModel Interface)

모든 모델은 `src/models/base_model.py`의 `BaseModel` 클래스를 상속받아야 합니다.
BaseModel은 4가지의 필수 추상 메소드를 정의합니다.

### 1-1. `calc_loss(self, batch_data)`

학습 시 Trainer가 호출하는 핵심 메소드입니다.
가장 중요한 점은 반환값이 **튜플(Tuple) 구조**를 따라야 한다는 것입니다.

```python
def calc_loss(self, batch_data):
    """
    Args:
        batch_data (dict): DataLoader가 제공하는 배치 데이터
                            {'user_id': ..., 'pos_item_id': ..., ...}

    Returns:
        (losses, params_to_log):
            - losses (tuple): (메인 로스, 보조 로스 1, 보조 로스 2, ...)
            - params_to_log (dict): {'log_name': value, ...}
    """
```

**튜플 반환 규약 (Tuple Protocol):**

- **Trainer의 처리 방식**: `Trainer`는 반환된 `losses` 튜플의 **모든 요소를 합산(`sum()`)**하여 `backward()`를 수행합니다.
- **첫 번째 요소**: 관습적으로 첫 번째 요소는 Main Loss(예: BPR, CrossEntropy)를 둡니다.
- **나머지 요소**: 정규화 항(L2, Orthogonal Loss 등)이나 보조 로스(Auxiliary Loss)를 둡니다.
- **가중치 적용**: 만약 보조 로스에 가중치($\lambda$)를 주고 싶다면, 여기서 곱해서 반환해야 합니다. `Trainer`는 단순히 합치기만 합니다.

**예시:**

```python
# CSAR_Sampled.py
def calc_loss(self, batch_data):
    # ... 계산 ...
    loss = self.loss_fn(...)
    orth_loss = self.attention_layer.get_orth_loss(...)

    # 가중치(lambda)는 여기서 곱해서 튜플로 포장
    return (loss, self.lamda * orth_loss), {'scale': ...}
```

### 1-2. `forward(self, users)` (Evaluation - Top-K)

- **목적**: `users`에 대해 **모든 아이템**의 점수를 예측합니다.
- **사용처**: 검증(Validation) 및 테스트 단계에서 Top-K Metric(NDCG, HitRate 등) 계산 시 사용.
- **주의**: 메모리 효율성을 위해 대규모 배치보다는 적절한 크기로 나누어 호출됩니다.

### 1-3. `predict_for_pairs(self, user_ids, item_ids)` (Evaluation - Pointwise)

- **목적**: 특정 `(user, item)` 쌍에 대해서만 점수를 예측합니다.
- **사용처**: 특정 분석이나 효율적인 검증이 필요할 때 사용될 수 있습니다.

### 1-4. `get_final_item_embeddings(self)`

- **목적**: 학습이 끝난 후 최종 아이템 임베딩 행렬을 반환합니다.
- **사용처**: `ItemKNN` 분석이나 임베딩 시각화, 혹은 `GiniIndex_emb`(임베딩 다양성) 지표 계산에 사용됩니다.

---

## 2. 새로운 모델 추가하기 (How to Add a New Model)

새로운 모델을 프레임워크에 추가하는 절차는 다음과 같습니다.

1.  **모델 파일 생성**: `src/models/{category}/` 디렉토리에 새 파이썬 파일(예: `my_model.py`) 생성.
2.  **BaseModel 상속**: 위에서 설명한 `BaseModel`을 상속받고 필수 메소드 4개를 구현.
3.  **레지스트리 등록**:

    - `src/models/__init__.py` 파일을 엽니다.
    - `my_model.py`를 import 합니다.
    - `MODEL_REGISTRY` 딕셔너리에 `'my-model-name': MyModelClass` 형태로 등록합니다.

    ```python
    # src/models/__init__.py
    from .general.my_model import MyModel

    MODEL_REGISTRY = {
        # ...
        'my-new-model': MyModel,
    }
    ```

4.  **설정 파일 생성**:
    - `configs/model/{category}/my_model.yaml`을 생성합니다.
    - `model.name` 필드는 레지스트리에 등록한 키 값(`'my-new-model'`)과 일치해야 합니다.

---

## 3. 손실 함수 (Loss Function) 확장

새로운 손실 함수는 `src/loss.py`에 자유롭게 추가할 수 있습니다.

- **규약**: `nn.Module`을 상속받는 것이 좋습니다.
- **입력**: 모델의 출력(Score/Logit)과 정답(Target)을 받습니다.
- **활용**: 모델의 `__init__`에서 인스턴스화하고, `calc_loss` 내부에서 호출합니다.

---

## 4. Trainer 내부 동작 원리

`Trainer`(`src/trainer.py`)는 다음과 같은 순서로 작동합니다.

1.  **초기화**: `config`를 파싱하여 Model, DataLoader, Optimizer 초기화.
2.  **학습 루프 (`train`)**:
    - `epoch` 순회
    - `DataLoader`로부터 배치 수신 (`user_id`, `pos_item_id` 등)
    - **Hard Negative Mining** (옵션): `use_hard_negatives: True`인 경우, `model.sample_hard_negatives`를 호출하여 `neg_item_id`를 덮어씀.
    - `model.calc_loss(batch)` 호출 -> `(losses_tuple, logs)` 수신
    - `final_loss = sum(losses_tuple)` 계산
    - `final_loss.backward()` 및 `optimizer.step()`
3.  **로깅 및 저장**: 학습 중 발생한 Loss와 `params_to_log`를 기록하고, `best_metric` 갱신 시 `trained_model/.../best_model.pt` 저장.
4.  **최종 평가**: 학습 종료 후 `final_method` 설정에 따라 전체 테스트셋 평가 수행.
