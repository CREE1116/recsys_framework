# RecSys Framework 🚀

최신 **협업 필터링(Collaborative Filtering)** 및 **딥러닝 기반 추천 모델**들의 성능을 일관된 환경에서 비교, 평가, 그리고 최적화하기 위해 구축된 통합 프레임워크입니다.

특히 **선형 공분산 필터링(Linear Covariance Filtering)**, **스펙트럼 편향 보정(Spectral Bias Correction)** 등 최신의 스펙트럼 기반 모델 라인업(CSAR, LIRA 계열)을 심층적으로 다루며, 전통적 베이스라인(MF, ItemKNN, EASE, LightGCN 등)과의 정확한 비교를 지원합니다.

---

## ✨ 핵심 기능

- **일관된 평가 프로토콜 (Unified Evaluation Protocol)**:
  - 엄격한 Leave-One-Out (시간순) 및 다양한 데이터 분할(Ratio, Random) 지원
  - 정교한 Negative Sampling 기법 내장 (RecBole 호환성 확보)
  - 정확도(NDCG, HitRate) 뿐만 아니라, 다양성(Coverage, ILD), 공정성(GiniIndex), **LongTail 제어 지표**까지 종합적으로 평가
- **효율적인 일괄 하이퍼파라미터 최적화 (Batch HPO)**:
  - Optuna 기반의 베이지안 최적화 내장
  - 다중 데이터셋 × 다중 모델 × 다중 시드(Multi-seed) 탐색을 YAML 설정 하나로 완벽 제어
- **모듈화된 아키텍처**:
  - 데이터 로딩, 손실 함수(Loss), 평가(Metrics), 모델 구조가 철저히 분리
  - 새로운 모델 추가가 매우 용이함 ([개발자 가이드](docs/DEVELOPER_GUIDE.md) 참조)
- **Closed-form 모델 네이티브 지원**:
  - 학습(SGD)이 필요 없는 선형 모델(EASE, LIRA 등)을 위한 전용 파이프라인 탑재

---

## 🏗 아키텍처 요약

```text
recsys_framework/
├── configs/            # YAML 기반 통합 설정 관리 (Evaluation, Dataset, Model)
├── data/               # Raw 데이터셋 저장소 (ml-100k, ml-1m, gowalla, yelp 등)
├── data_cache/         # 전처리 완료된 데이터 객체 캐싱 (빠른 재로딩)
├── docs/               # 세부 문서 모음 (아키텍처, 손실 함수, 평가 프로토콜 등)
├── output/             # 베이지안 최적화 및 일괄 실험 결과, 모델별 리포트 저장소
├── scripts/            # 실험 실행, HPO 탐색, 데이터 전처리 매니저 스크립트 모음
├── src/                # 프레임워크 핵심 코어
│   ├── losses/         # BPR, InfoNCE, Pointwise Loss 등 다양한 손실함수
│   ├── models/         # 추천 모델 구현체 모음 (General, CSAR)
│   ├── trainer.py      # 학습(Train), 검증(Valid), 평가(Test) 루프 오케스트레이션
│   └── evaluation.py   # Top-K 랭킹 및 10가지 이상의 상세 메트릭 계산
```

---

## 📦 구현된 모델 라인업

### 1. CSAR & 스펙트럼 모델 계열 (Ours)

본 프레임워크의 핵심 연구 대상 모델들입니다. ([CSAR 모델 요약 문서](docs/csar_models_summary.md))

- **ASPIRE**: MNAR 편향의 스펙트럼 서명을 측정하고(SWLS), 이를 Closed-form 영역에서 수정하는 스펙트럼 보정 필터링 모델 (SVD 기반).
- **ChebyASPIRE**: 거대한 데이터셋 동작을 위해 SVD 대신 쳬비쇼프(Chebyshev) 다항식 근사를 사용한 ASPIRE의 선형-시간(Linear-time) 스케일링 버전.
- **AspireBPR**: ASPIRE의 스펙트럼 페널티 보정 원리를 SGD 기반의 사용자-아이템 행렬 분해(BPR 학습)에 이식한 신경망 모델.
- **LIRA (Linear Representation Alignment)**: 타겟 임베딩과 유사도 행렬의 주파수 대역을 맞추는 선형 보정 필터.
- **CSAR (Co-Support Attention RecSys)**: 관심사(Latent Topics) 기반의 에너지 점수를 사용하는 해석 가능한 추천 모델.

### 2. General Baselines

최신 논문 비교를 위한 최고 수준의 최적화가 적용된 베이스라인들입니다. ([General 모델 요약 문서](docs/general_models_summary.md))

- **EASE (Embarrassingly Shallow Autoencoders)**: 희소 데이터에 매우 강력한 Non-SGD Closed-form 선형 베이스라인 모델.
- **LightGCN**: 최신의 Graph Convolutional Network (GCN) 기반 추천시스템 SOTA 베이스라인.
- **MF / NeuMF**: 행렬 분해 기반 기본 모델과 다층 퍼셉트론 혼합 모델.
- **MultVAE**: Multinomial Likelihood를 최적화하는 생성형 오토인코더 베이스라인.
- **ProtoMF / ACF**: 앵커/프로토타입을 통해 임베딩을 구성하여 다양성을 모색하는 모델.
- **GF-CF, RLAE, SANSA** 등 최신 스펙트럼 및 선형 CF 비교 기술들.

---

## 🏃 시작하기

### 1. 환경 설정

본 프로젝트는 의존성 관리자로 `uv`의 사용을 적극 권장합니다.
Python 3.12 이상의 환경에서 실행하세요.

```bash
uv pip install -r docs/requirements.txt
```

### 2. 단일 모델 학습 및 평가

특정 데이터셋(예: ml-100k)과 특정 모델(예: LightGCN)을 즉시 학습하고 평가합니다.

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml100k.yaml \
  --model_config configs/model/general/lightgcn.yaml
```

### 3. 일괄 하이퍼파라미터 탐색 (Batch HPO)

여러 데이터셋과 모델에 대해 Optuna 베이지안 최적화를 한 번에 실행합니다. 검색 공간은 YAML 파일로 완벽하게 통제됩니다.

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/paper_baselines
```

---

## 📚 상세 문서

프레임워크의 내부 구조나 추가 세부 사항을 파악하려면 `docs/` 디렉토리의 문서들을 참고하세요:

1. [Configuration Guide (`CONFIG.md`)](docs/CONFIG.md): 3단계 병합 시스템과 YAML 파라미터 구조 안내
2. [Evaluation Protocol (`EVALUATION_PROTOCOL.md`)](docs/EVALUATION_PROTOCOL.md): LOO 기반 정밀 평가, 마스킹 전략 및 메트릭 해설
3. [Developer Guide (`DEVELOPER_GUIDE.md`)](docs/DEVELOPER_GUIDE.md): 새로운 모델 추가, Loss 정의, 데이터 파이프라인 아키텍처 가이드
4. [Loss Functions (`loss_functions_summary.md`)](docs/loss_functions_summary.md): BPR, MSE, InfoNCE(Sampled Softmax) 등 프레임워크 내장 Loss 수식

---

_Developed for research on advanced Recommender System methodologies._
