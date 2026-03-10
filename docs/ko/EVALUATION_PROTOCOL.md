# 평가 프로토콜

## 데이터 분할

분할 방식은 데이터셋 설정의 `split_method`로 지정합니다. 평점 임계값, k-core 필터링 등 모든 전처리 옵션도 설정에서 구성하며 하드코딩된 값이 아닙니다.

### 분할 방식

**`loo` (Leave-One-Out):**

유저별 마지막 상호작용(타임스탬프 기준, 동점 시 item_id)을 테스트 세트로, 두 번째 마지막을 검증 세트로, 나머지를 학습 세트로 사용합니다. 유저당 테스트 아이템이 정확히 1개이므로 `HitRate@K == Recall@K`입니다.

**`temporal_ratio` (alias: `temporal`):**

유저별 상호작용을 시간순 정렬 후, 마지막 `test_ratio` 비율을 테스트, 다음 `valid_ratio` 비율을 검증, 나머지를 학습으로 사용합니다. 타임스탬프 컬럼이 필요합니다. 유저당 여러 테스트 아이템이 생길 수 있습니다.

**`random`:**

`temporal_ratio`와 동일하지만 타임스탬프 정렬 없이 랜덤으로 분할합니다. 재현성을 위해 `seed`를 설정할 수 있습니다. 타임스탬프가 없는 데이터셋에 사용합니다.

**`presplit`:**

LightGCN 형식(`user_id item_id item_id ...`)의 외부 학습/테스트 파일을 로드합니다. k-core 필터링이나 ID 리매핑을 적용하지 않으며, ID가 이미 전처리되어 0부터 시작한다고 가정합니다. `train_file`과 `test_file` 설정 필드가 필요합니다. 검증 세트는 생성되지 않습니다.

### 전처리 옵션

다음 전처리 옵션은 `presplit`을 제외한 모든 분할 방식에 적용됩니다:

| 옵션 | 설정 키 | 기본값 | 설명 |
|---|---|---|---|
| 평점 임계값 | `rating_threshold` | `null` (전체) | 이 값 이상의 평점만 positive로 사용 |
| 최소 유저 상호작용 | `min_user_interactions` | `5` | 상호작용이 적은 유저 제외 (k-core) |
| 최소 아이템 상호작용 | `min_item_interactions` | `5` | 상호작용이 적은 아이템 제외 (k-core) |
| 중복 제거 | `dedup` | `true` | 동일 (유저, 아이템) 쌍 중복 제거 |

---

## 네거티브 샘플링

| 설정 | 값 | 설명 |
|---|---|---|
| 제외 대상 | Train 이력만 | 테스트 아이템도 negative로 등장 가능 (RecBole 표준) |
| 샘플링 전략 | 독립 균등 샘플링 | 배치 내 후보 풀 공유 없음 |

---

## 평가 방법

| 방법 | 설명 |
|---|---|
| `full` | 전체 아이템 스코어링 → train 이력 마스킹 → Top-K 추출 |
| `sampled` | 랜덤 유저 부분집합에 대해 전체 아이템 스코어링 |
| `uni99` | 유저당 1개 positive + 99개 균등 랜덤 negative |

테스트 시 train 및 validation 상호작용은 랭킹에서 마스킹됩니다. 테스트 타겟 자체는 마스킹하지 않습니다.

---

## 메트릭

**정확도:**
- `NDCG@K` — 정규화 할인 누적 이득
- `HitRate@K` — 테스트 아이템이 Top-K에 있으면 1
- `Recall@K` — Top-K에 포함된 테스트 아이템 비율 (LOO에서는 HitRate와 동일)
- `Precision@K`
- K 값: 5, 10, 20, 50

**다양성 및 공정성:**
- `Coverage@K` — 적어도 한 번 이상 추천된 아이템 비율
- `ILD@K` — 리스트 내 다양성 (아이템 임베딩 기반 평균 쌍별 거리)
- `GiniIndex@K` — 아이템 추천 빈도의 지니 계수
- `LongTailCoverage@K` — 인기도 하위 20% 아이템에 대한 커버리지
- `LongTailRatio@K` — 추천 중 롱테일 아이템의 비율

**신규성:**
- `Novelty@K` — 자기정보 기반 신규성 (덜 인기 있는 아이템일수록 높음)

---

## 기본 하이퍼파라미터

| 파라미터 | 값 |
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

유저당 테스트 아이템이 여러 개인 경우(비율 기반 분할)에만 차이가 발생합니다:

| | `loo` | `temporal_ratio` / `random` |
|---|---|---|
| 유저당 정답 수 | 1개 | N개 (가변) |
| HitRate@K | 테스트 아이템이 Top-K에 있으면 1 | 테스트 아이템 중 하나라도 Top-K에 있으면 1 |
| Recall@K | HitRate와 동일 | (Top-K에 포함된 테스트 아이템 수) / (전체 테스트 아이템 수) |
| NDCG@K | `1/log2(rank+1)` | 다중 아이템 DCG/IDCG |
