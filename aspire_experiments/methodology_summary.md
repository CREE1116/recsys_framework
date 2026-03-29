# ASPIRE Experiments: Methodology Summary (Refined v4)

본 문서는 ASPIRE 프레임워크의 이론적 검증 및 성능 분석을 위한 주요 실험들의 방법론을 정리합니다. (결과 및 결론 제외)

## 실험 1: Spectral Power-law (SPL) Verification
- **목적**: 특이값($\sigma_k$)과 스펙트럼 관측 확률($p_k$) 사이의 로그-로그 선형 관계(Power-law)를 엄밀히 검증.
- **방법**: 전체 에너지의 95%를 포함하는 성분들에 대해 LAD(Least Absolute Deviations) 피팅을 수행하고, 지수 분포(Exponential) 대비 파워-법칙(Power-law)의 적합도를 LL-Ratio 및 K-S Statistic으로 측정.

## 실험 2: Spectral Symmetry & Gap Analysis
- **목적**: Bridge Lemma 기반의 이론적 $\gamma$값과 실제 최적(Empirical Best) $\gamma$값 사이의 간극을 분석.
- **방법**: 데이터셋별 Zipf 지수($\zeta$)를 추정하여 이론적 $\gamma = 2-\zeta$를 산출하고, 원본 스펙트럼, 이론적 보정 스펙트럼, 실제 최적 보정 스펙트럼의 로그-로그 기울기를 비교.

## 실험 3: Debiasing & Popularity Analysis
- **목적**: 인기도 편향 제거 능력과 추천 정확도 사이의 균형을 분석.
- **방법**: 아이템을 인기도 순으로 10개 구간(Decile)으로 나누어, 각 모델(ASPIRE, EASE, LIRA)별로 구간당 추천 빈도(Rec Proportion)와 적중률(HitRate@20)을 측정.

## 실험 4: ChebyASPIRE Efficiency & Trade-off
- **목적**: Full SVD 방식과 차수($N$)별 Chebyshev 근사 방식의 효율성 및 성능 비교.
- **방법**: 차수를 1에서 40까지 변화시키며 학습 시간(Build time), 추론 시간(Eval time), 최대 메모리 사용량(Peak memory), 그리고 정확도(NDCG@20)의 변화 추이를 측정.

## 실험 5: Accuracy-Coverage Trade-off Analysis
- **목적**: ASPIRE와 기존 탈편향 모델(IPS-LAE) 및 베이스라인(EASE) 간의 성능-다양성 트레이드오프 비교.
- **방법**: $\gamma$ (ASPIRE) 및 $w_\beta$ (IPS-LAE) 파라미터를 스윕(Sweep)하며 Coverage@20 대비 NDCG@20의 변화를 곡선으로 시각화하고, 인기 아이템(Head)과 롱테일(Tail) 구간의 정확도를 별도 측정.

## 실험 6: Popularity vs. Rec Frequency Visualization
- **목적**: 아이템 인기도와 추천 빈도 사이의 상관관계를 로그-로그 스케일에서 시각화.
- **방법**: 각 모델별 최적 하이퍼파라미터를 찾은 후(HPO), 아이템별 인기도별 평균 추천 빈도를 로그-로그 평면에 플로팅하여 편향 완화 정도를 직관적으로 분석.

## 실험 7: Sparsity Robustness Analysis
- **목적**: 데이터의 희소성(Sparsity)이 모델 성능에 미치는 영향 분석.
- **방법**: 전체 학습 상호작용의 일부(20%, 40%, 60% 등)를 무작위로 제거(Masking)한 후, 데이터 가용성 변화에 따른 각 모델의 NDCG@20 유지 능력을 평가.

## 실험 8: Gamma-Alpha Ablation Study
- **목적**: 필터 함수의 구성 요소($\alpha, \gamma$) 및 이론적 추정 방식의 기여도 분석.
- **방법**: 필터 모델을 4가지 설정(이론적 $\gamma$ 고정, 최적 $\gamma$ 탐색, $\alpha$ 유무 등)으로 나누어 비교 평가하고, Overall NDCG뿐만 아니라 Coverage와 Tail-NDCG를 함께 정량화.

## 실험 9: Beta-Gamma Stability Analysis
- **목적**: 최상위 특이값(Goliath components)이 이론적 파라미터 추정 안정성에 미치는 영향 분석.
- **방법**: 최상위 특이값($\sigma_1 \sim \sigma_k$)을 의도적으로 제외(Skip)하며 HPO를 수행하고, 이 과정에서 도출되는 최적 $\gamma$와 이론적 예측치 간의 수렴 여부를 분석.

## 실험 15: Spectral Structure Proof (Master)
- **목적**: "스펙트럴 멱법칙(SPL)은 임의적 가정이 아니라 추천 피드백 루프의 구조적 산물이다"라는 논문의 핵심 논리를 시뮬레이션과 실제 데이터를 통해 통합 증명.
- **방법**: 
    - **Panel A (Simulation)**: 무작위 초기 행렬에서 Top-K Softmax 피드백 루프를 반복하여, 구조가 없던 데이터에서 SPL($R^2$ 상승)이 자생적으로 출현하는 과정을 시각화.
    - **Panel B (Real Data)**: Yahoo! R3 데이터셋의 MNAR(Train)과 MCAR(Test) 데이터를 직접 비교하여, 편향이 존재하는 데이터에서만 가파른 SPL 기울기가 나타남을 입증.
- **분석**: LAD(Least Absolute Deviations) 피팅을 통해 특이값($\sigma_k$)과 스펙트럴 인기도($p_k$) 사이의 로그-로그 선형성을 정량화.
