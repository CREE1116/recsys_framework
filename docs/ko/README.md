# RecSys Framework

추천 시스템 연구를 위한 경량 프로토타이핑 프레임워크입니다. 새로운 모델을 빠르게 구현하고 검증할 수 있도록 설계되었으며, CUDA 및 Apple Silicon(MPS) GPU 가속을 기본으로 지원합니다.

---

## 개요

주요 기능:

- **디바이스 자동 감지 GPU 가속** — `CUDA → MPS → CPU` 순으로 자동 폴백. SVD, Gram 행렬 풀이, Cholesky 분해 등 모든 무거운 선형 대수 연산에 적용됩니다.
- **SVD 및 Gram Eigen 캐싱** — 한 번 계산한 분해 결과를 디스크에 저장하고 재사용합니다. 캐시 히트 시 로딩이 즉각적으로 이루어집니다.
- **YAML 기반 설정** — `evaluation → dataset → model` 3단계 설정 파일로 파이프라인 전체를 제어합니다. 코드 수정이 필요 없습니다.
- **단순한 모델 인터페이스** — `BaseModel`이 최소 4개 메서드만 요구합니다. Closed-form 모델은 `fit()`, 경사하강 모델은 `calc_loss()`를 구현합니다.
- **일괄 HPO** — Optuna 기반 베이지안 탐색으로 여러 모델 × 여러 데이터셋 조합을 멀티시드로 자동 실험합니다.
- **종합적인 평가** — Full ranking 및 Sampled 평가, 정확도·다양성·신규성·롱테일 메트릭 지원.

---

## 디렉터리 구조

```
recsys_framework/
├── configs/                  # YAML 설정 (evaluation, dataset, model)
├── data/                     # 원시 데이터셋 파일
├── data_cache/               # 전처리 데이터 및 SVD/Gram 캐시 (자동 관리)
├── docs/                     # 기술 문서 (영문)
│   └── ko/                   # 기술 문서 (한글)
├── output/                   # HPO 탐색 결과 및 평가 로그
├── scripts/
│   ├── main.py               # 단일 실험 실행 진입점
│   └── run_all_smart_searches.py  # 일괄 HPO 실행기
└── src/
    ├── utils/
    │   ├── gpu_accel.py      # GPU 선형 대수 가속 (SVD, Cholesky, Gram)
    │   └── cache_manager.py  # 캐시 생애주기 관리
    ├── models/
    │   ├── base_model.py     # 추상 기반 클래스
    │   ├── general/          # 베이스라인 모델 (EASE, LightGCN, MF, ...)
    │   └── csar/             # LIRA 및 ASPIRE 계열
    ├── data_loader.py        # 데이터 로딩, 필터링, train/val/test 분할
    ├── evaluation.py         # Top-K 랭킹 메트릭
    ├── trainer.py            # 학습 및 평가 오케스트레이터
    └── loss.py               # BPR, SampledSoftmax, MSE, DynamicMarginBPR
```

---

## 빠른 시작

### 설치

```bash
uv pip install -r docs/requirements.txt
```

Python 3.12 이상 권장. 패키지 관리는 `uv` 사용을 권장합니다.

### 단일 실험 실행

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml1m.yaml \
  --model_config configs/model/general/ease.yaml
```

### 일괄 HPO 실행

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/paper_baselines
```

---

## 구현된 모델

### Closed-form (경사하강 불필요)

| 모델 | 설명 |
|---|---|
| `ease` | EASE — closed-form ridge regression으로 아이템-아이템 가중치 행렬 계산 |
| `lira` | LIRA — 유저-유저 Gram 풀이를 통한 dual ridge regression |
| `light_lira` | LightLIRA — SVD 기반 스펙트럼 근사 LIRA |
| `aspire` | ASPIRE — MNAR 보정을 적용한 인기도 디바이어스 아이템 유사도 |
| `item_knn` | 아이템 기반 코사인 KNN |
| `most_popular` | 인기도 기반 비개인화 추천 (하한 기준점) |
| `pure_svd` | 절단 SVD 기반 추천 |
| `svd_ease` | SVD 근사 EASE |
| `slim` | SLIM — 희소 선형 모델 |

### 경사하강 기반

| 모델 | 설명 |
|---|---|
| `mf` | Matrix Factorization (BPR/Softmax 손실) |
| `neumf` | Neural MF (GMF + MLP) |
| `lightgcn` | LightGCN — 그래프 합성곱 협업 필터링 |
| `multivae` | 암묵적 피드백을 위한 다항 VAE |
| `protomf` | 직교성 정규화를 갖는 프로토타입 기반 MF |
| `ultragcn` | UltraGCN — 제약 기반 그래프 CF |

---

## 문서

- **[개발자 가이드](DEVELOPER_GUIDE.md)** — 모델 추가 방법, BaseModel 인터페이스, Trainer 흐름
- **[설정 가이드](CONFIG.md)** — YAML 설정 시스템, 전체 파라미터 설명
- **[GPU 가속](GPU_ACCEL.md)** — SVDCacheManager, Cholesky 풀이, Gram 행렬 풀이, 디바이스 디스패치
- **[평가 프로토콜](EVALUATION_PROTOCOL.md)** — 데이터 분할, 네거티브 샘플링, 메트릭
- **[모델 레퍼런스](general_models_summary.md)** — 구현된 모델 설명
- **[손실 함수](loss_functions_summary.md)** — BPR, SampledSoftmax, MSE

---

## 디바이스 지원

모든 연산 집약적 작업은 자동으로 디스패치됩니다:

```
CUDA (사용 가능 시) → MPS (Apple Silicon, 사용 가능 시) → CPU
```

SVD 계산, Cholesky 분해, Gram 행렬 고유값 분해에 동일하게 적용됩니다. CUDA의 대규모 행렬 SVD는 희소 CSR 텐서 위에서 네이티브 랜덤화 알고리즘(`torch.sparse.mm` + `torch.linalg.qr`)을 사용합니다.
