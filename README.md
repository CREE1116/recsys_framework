# RecSys Framework 🚀

Recommender System 연구자들을 위해 설계된 **경량 & 고속 프로토타이핑 프레임워크**입니다.
복잡한 엔지니어링 뎁스를 최소화하여 새로운 수식이나 모델을 즉시 구현하고 테스트할 수 있는 연구 환경을 제공합니다.

특히 최신 **Apple Silicon (MPS)**을 적극 지원하여 로컬 환경에서 대용량 SVD 및 행렬 연산을 원활하게 수행할 수 있도록 최적화되었습니다.

---

## ✨ 핵심 철학 및 기능

- **Mac(MPS) 친화적 GPU 가속**:
  - `gpu_accel.py`를 통해 MPS(Metal Performance Shaders)와 CUDA를 완벽히 지원.
  - 대규모 행렬 분해(SVD), Cholesky Solver 등 병목이 되는 선형 대수 연산을 GPU 기반 텐서 연산으로 이관하여 속도 극대화.
- **연구를 위한 극강의 개발 편의성**:
  - **Auto-Logging**: `trainer.py`가 에폭마다 메트릭을 자동 로깅하며 텐서보드 설정 불필요.
  - **YAML-driven Architecture**: 하드코딩 없이 세팅 파일만으로 모든 파이프라인 통제.
  - **Easy Model Extension**: `BaseModel` 클래스의 `calc_loss()` 4줄만 수정하면 새로운 추천 모델 즉시 완성.
- **스마트 글로벌 캐시 매니저 (Global Cache Manager)**:
  - 수십 분이 걸리는 대용량 SVD나 고유값 분해(Eigen Decomposition), 전처리된 데이터셋 등을 `CacheManager`가 글로벌하게 캐싱.
  - 동일한 데이터나 파라미터 조합 재실행 시 계산을 100% 생략하여 **"0-초" 로딩** 실현.
- **일괄 하이퍼파라미터 최적화 (Batch HPO)**:
  - 다중 데이터셋 × 다중 베이스라인 모델을 Optuna 기반 베이지안 서치로 며칠에 걸쳐 한 번에 탐색하고 종합 리포트를 남기는 매크로 스크립트 보유.

---

## 🏗 아키텍처 요약

```text
recsys_framework/
├── configs/            # YAML 기반 설정 (Evaluation, Dataset, Model, Search)
├── data/               # Raw 데이터셋 저장소
├── data_cache/         # SVD 결과, 그래프 구성 등 리소스 집중 연산의 자동 캐싱
├── docs/               # 개발자 가이드, 구조 요약 문서 (하단 링크 참조)
├── output/             # 베이지안 최적화 및 평가 결과 JSON / 로그
├── scripts/            # 실험 실행(main.py) 및 HPO 탐색 매니저
├── src/
│   ├── utils/
│   │   ├── gpu_accel.py       # MPS/CUDA 가속 선형 대수 라이브러리 (SVD, Cholesky)
│   │   └── cache_manager.py   # 전역 메모리/디스크 캐시 관리자
│   ├── data_loader.py         # 자동 데이터 전처리 및 Split (LOO, Random 등)
│   ├── evaluation.py          # Top-K 랭킹 및 다양성/롱테일 메트릭 동시 계산
│   └── trainer.py             # 오케스트레이션 및 자동 저장 루프
```

---

## 🏃 시작하기

### 1. 환경 설정

`uv` 패키지 매니저 사용을 권장합니다. (Python 3.12+ 지원)

```bash
uv pip install -r docs/requirements.txt
```

### 2. 단일 프로토타입 즉시 실행

YAML 옵션을 결합해 즉각적인 검증을 돌립니다. (MPS 가속 기본 적용)

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml100k.yaml \
  --model_config configs/model/general/lightgcn.yaml
```

### 3. 무인 HPO 탐색 (Batch Search)

여러 모델과 데이터셋을 YAML 하나에 정의해두고 베이지안 최적화를 넘깁니다.

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/paper_baselines
```

---

## 📚 기술 문서 가이드 (Docs)

디자인 패턴과 구현에 대한 자세한 내용은 아래 문서를 읽어보세요. 연구용으로 커스텀이 필요할 때 가장 먼저 읽어야 할 자료들입니다.

1. **[Developer Guide (`docs/DEVELOPER_GUIDE.md`)](docs/DEVELOPER_GUIDE.md)**
   - "나만의 새 모델 만드는 법"
   - `BaseModel` 인터페이스, Loss 등록, Trainer 작동 원리가 요약되어 있습니다.
2. **[Configuration Guide (`docs/CONFIG.md`)](docs/CONFIG.md)**
   - evaluation → dataset → model 순으로 덮어써지는 3-Tier YAML 파라미터 구조 안내.
3. **[Evaluation Protocol (`docs/EVALUATION_PROTOCOL.md`)](docs/EVALUATION_PROTOCOL.md)**
   - LOO 분할 규칙, Negative Sampling(InfoNCE 포함), 그리고 NDCG부터 LongTailCoverage까지 지원되는 메트릭의 정의.
4. **[Loss & Metrics Summary (`docs/loss_functions_summary.md`)](docs/loss_functions_summary.md)**
   - 구현된 손실함수 수식과 언제 어떤 Loss를 써야 유리한지 정리.
5. **[Model Inventory](docs/)**
   - 개발된 모델 라인업의 이론적 요약 ([`csar_models_summary.md`](docs/csar_models_summary.md), [`general_models_summary.md`](docs/general_models_summary.md))

---

_Built for fast iteration, solid evaluation, and cutting-edge RecSys theory experiments on Apple Silicon._
