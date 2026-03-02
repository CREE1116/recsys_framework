# RecSys Framework

PyTorch 기반의 **연구용 추천시스템 프레임워크**입니다. 빠른 프로토타이핑과 공정한 벤치마크를 목표로 합니다.

---

## 🌟 주요 기능

- **50+ 모델**: MF, LightGCN, Multi-VAE, EASE, UltraGCN, GF-CF, LIRA 시리즈 등
- **통합 파이프라인**: `Trainer.run()` 하나로 fit → train → evaluate 자동 분기
- **3가지 평가 프로토콜**: Full ranking, Uni99, Sampled validation
- **15+ 평가 메트릭**: NDCG, HitRate, Recall, Precision, Coverage, ILD, Novelty, GiniIndex, PopRatio, LongTail 시리즈
- **Bayesian HPO**: Optuna 기반, 멀티시드 평균, 자동 체크포인트 관리
- **AMP 자동 적용**: CUDA/MPS Mixed Precision 가속

---

## 🆚 RecBole 대비 강점

| Feature             | This Framework                                             | RecBole                         |
| :------------------ | :--------------------------------------------------------- | :------------------------------ |
| **새 모델 추가**    | 파일 1개 + 등록 1줄                                   | Config/Data 연동 복잡    |
| **Deep Inspection** | `calc_loss` 튜플로 Loss 항목별 자동 추적 + PNG 그래프 생성 | 단일 scalar, 커스텀 로깅 어려움 |
| **Loss 유연성**     | 튜플 반환 → Main/Reg 분리 추적 가능                        | 단일 loss 강제                  |
| **커스터마이징**    | Python 그대로, 숨겨진 로직 없음                            | 방대한 추상 클래스 상속 구조    |

> **"연구자는 프레임워크와 싸우지 말고, 모델링에 집중해야 합니다."**

---

## 🧩 모델 라인업

### General Baselines

| 카테고리                 | 모델                                                                                       |
| :----------------------- | :----------------------------------------------------------------------------------------- |
| **Matrix Factorization** | MF, NeuMF, SoftplusMF, iALS, ProtoMF                                                       |
| **Graph**                | LightGCN, UltraGCN, SimGCL, GF-CF                                                          |
| **AutoEncoder**          | Multi-VAE, EASE, NormEASE, RLAE, SLIM, ELSA, SANSA, SVD-AE, SVD-EASE, NC-EASE, Infinity-AE |
| **Others**               | ItemKNN, PureSVD, MostPopular, MACR, MMR, NaiveBayes, CoOccurrence                         |
---

## ⚙️ 설정 시스템

모든 실험은 YAML 파일로 제어됩니다. 3단계 병합: `evaluation.yaml`(기본) → dataset(데이터셋 특화) → model(최종).

```
configs/
├── evaluation.yaml           # 평가 마스터 설정 (메트릭, 프로토콜)
├── dataset/                  # 데이터셋별 설정
│   ├── ml100k.yaml
│   ├── ml1m.yaml
│   └── ...
├── model/                    # 모델별 하이퍼파라미터
│   ├── general/              # 베이스라인 모델
│   └── csar/                 # CSAR/LIRA 모델
└── paper_baselines_search.yaml  # HPO 설정
```

### 예시: 모델 설정

```yaml
model:
  name: "ease"
  reg_weight: 500.0

# train 블록이 없으면 비학습 모델 → fit() 후 바로 평가
```

```yaml
model:
  name: "lightgcn"
  embedding_dim: 64
  n_layers: 3

train:
  batch_size: 1024
  epochs: 100
  lr: 0.001
  loss_type: "pairwise"
  embedding_l2: 1.0e-5
```

---

## 🚀 시작하기

### 환경 설정

```bash
uv venv --python 3.12.0
source .venv/bin/activate
uv pip install -r docs/requirements.txt
```

### 단일 모델 실행

```bash
cd scripts/
uv run python main.py \
  --dataset_config ../configs/dataset/ml100k.yaml \
  --model_config ../configs/model/general/ease.yaml
```

### Bayesian HPO

```bash
cd scripts/
uv run python bayesian_opt.py \
  --dataset_config ../configs/dataset/ml100k.yaml \
  --model_config ../configs/model/general/lightgcn.yaml
```

### 전체 벤치마크

```bash
cd scripts/
uv run python run_all_smart_searches.py \
  --config ../configs/paper_baselines_search.yaml \
  --output_dir ../output/paper_baselines
```

---

## 📊 디렉토리 구조

```
.
├── configs/                  # YAML 설정 파일
├── data/                     # 데이터셋 저장소
├── scripts/                  # 실행 스크립트
│   ├── main.py               #   단일 실험 엔트리포인트
│   ├── bayesian_opt.py       #   Bayesian HPO
│   └── run_all_smart_searches.py  # 전체 벤치마크
├── src/                      # 프레임워크 코어
│   ├── data_processing.py    #   순수 함수: 로드, 필터, 리매핑, 분할
│   ├── data_loader.py        #   오케스트레이터 + 캐시 + 로더 팩토리
│   ├── trainer.py            #   run() → fit/train/evaluate 통합
│   ├── evaluation.py         #   15+ 메트릭 계산
│   ├── loss.py               #   BPR, MSE, SampledSoftmax, DynamicMarginBPR
│   └── models/
│       ├── base_model.py     #   BaseModel 추상 클래스
│       ├── general/          #   일반 베이스라인
│       └── csar/             #   CSAR/LIRA 연구 모델
├── analysis/                 # 실험 결과 분석 스크립트
├── trained_model/            # 실험 결과 (모델, 메트릭, 그래프)
├── docs/                     # 상세 문서
│   ├── CONFIG.md
│   └── EVALUATION_PROTOCOL.md
└── tests/                    # 유닛 테스트
```

---

## 📚 문서

모든 문서는 `docs/` 디렉토리에 있습니다.

- **[개발자 가이드](docs/DEVELOPER_GUIDE.md)**: BaseModel 인터페이스, Trainer 동작 원리, 모델 추가 방법
- **[설정 가이드](docs/CONFIG.md)**: YAML 설정 시스템 상세
- **[평가 프로토콜](docs/EVALUATION_PROTOCOL.md)**: Full/Uni99/Sampled 평가 방식 설명
- **[Loss 함수 정리](docs/loss_functions_summary.md)**: BPR, InfoNCE, MSE, DynamicMarginBPR 수식 및 용도
- **[General 모델 요약](docs/general_models_summary.md)**: 일반 베이스라인 모델 상세
- **[CSAR 모델 요약](docs/csar_models_summary.md)**: CSAR/LIRA 연구 모델 상세
