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

## 실험 17: Eigenvector Stability Analysis
- **목적**: MNAR 환경에서 승산적 인기도 편향(Multiplicative Popularity Bias)이 가해졌을 때, 실제 공분산과 관측된 공분산 행렬 사이의 주요 고유벡터 보존 여부를 분석.
- **방법**: 파워 법칙을 따르는 가상의 아이템 인기도 구조를 생성하고, 이에 편향 강도(`bias_strength`)를 적용하여 $C_{obs} = D C_{true} D$ 를 유도. 이후 Top-K 고유벡터의 1:1 코사인 유사도와 하위 공간(Subspace) 각도를 비교하여, 인기도 편향이 고유공간(Eigenspace) 방향성에 미치는 왜곡 정도를 정량화 및 시각화.

## 실험 18: ASPIRE Spectral Penalty & Eigenvector Invariance Analysis
- **목적**: ASPIRE의 실제 필터가 스펙트럼 에너지 분포(고유값)만을 재조정할 뿐 방향(고유벡터)을 변화시키는지 검증.
- **방법**: 이론적 편향 환경에서 관측된 왜곡 행렬($C_{obs}$)에 ASPIRE 고유 필터 수식($h(\lambda)$)을 적용하여 새로운 재구성 행렬($C_{aspire}$)을 산출. 이후 $C_{aspire}$에 재차 EVD 연산을 수행하여 생성된 새로운 고유벡터($V_{aspire}$) 가 원본 찌그러진 고유벡터($V_{obs}$)와 일치하는지, 즉 $V$ 공간의 회전이나 정답 공간($V_{true}$)으로의 복원이 일절 발생하지 않음을 기하학적 유사도로 증명. 동시에, $\gamma$의 역할이 거대한 엘리트 특이값 에너지를 억눌러 꼬리 성분의 발언권(영향력)을 해방시키는 에너지 평탄화(Flattening)라는 점을 플롯으로 가시화.

## 실험 19: IPS + Wiener vs ASPIRE 성능 한계 비교 기하학 검증
- **목적**: 노이즈가 낀 현실의 제약적 상황 하에서 고전적인 '확률 스케일링(IPS) 및 역행렬 정규화' 결합 모델이 겪는 분산 폭발의 한계와, 우회적 스펙트럴 패널티 모델인 ASPIRE의 안정성(Robustness) 및 정답 복원력을 충돌 대조 분석.
- **방법**: 잠재 편향 공간 $C_{obs}$ 에 인위적인 관측 노이즈 행렬 $E$를 섞어 현실 데이터 환경 모방. 이후 1) $D^{-1}$로 직접 역확률을 스케일링한 뒤 $\alpha$ 스윕의 Wiener Filter를 통과시킨 대조군과 2) 역스케일링 없이 스펙트럼 자체를 분수($\gamma$)로 찍어누른 ASPIRE 대조군의 $C_{true}$ 행렬 대비 매트릭스 도트 유사도를 산출. 꼬리 아이템의 무한대 역확률 때문에 파괴되는 IPS 방식의 붕괴점과 ASPIRE의 우월한 복원력을 오버레이 그래프로 검증.

## 실험 20: 랭킹 측면에서의 Proxy IPS vs ASPIRE 비교 (End-to-End NDCG 검증)
- **목적**: 이론적 고품질 오라클(Oracle) 확률 대신 거친 관측치 기반 '빈도 프록시(Empirical Popularity)'를 사용할 때 겪는 분산 폭발로 인해, 추천 시스템 핵심 지표인 NDCG 랭킹 성능에서 ASPIRE가 Proxy 전향성 기법에 대해 갖는 압도적인 한계점 우월성을 검증.
- **방법**: 유저-아이템 희소 이진 데이터 매트릭스(Sparse Binary Matrix)를 시뮬레이션하고 멱법칙 기반 MNAR 누락(편향)을 주입. 이를 학습 데이터로 삼아 **Proxy IPS 모델 (빈도수 프록시 역확률 스케일링 모델)**과 **ASPIRE 모델 (순수 스펙트럴 에너지 제어 모델)**의 스윕 예측 성능 $\hat{R}$ 을 산출한 뒤, Test 셋에 대해 추천 품질(NDCG@20)을 비교 시각화.

## 실험 21: 하이퍼파라미터($\gamma$) 이론적 무비용 자동 추정 기법 증명 (Zero-cost HPO) [추정 포기]
- **목적**: 무거운 병렬 탐색 컴퓨팅 자원이나 정답 셋 없이도 최적의 스펙트럴 제어강도 $\gamma$ 를 1초 만에 추정해 내는 휴리스틱 수학 탐색법들의 성능 비교.
- **방법**: 실제 데이터(MNAR) 모델을 통해 스윕 NDCG 곡선의 최고봉 즉, "HPO 정답 피크선(Ground Truth Peak)" 을 찾고, 1) 파레토 꼬리 기울기, 2) 스펙트럴 엘보우, 3) 스펙트럴 엔트로피 변곡점 등 수리적 추정선들과의 거리를 오버레이 검증.
- **결론**: 기하학적 형태만으로 추천 시스템의 User-Item 신호 보존율을 완벽히 계산할 수 없으므로, 추정값들($\gamma \approx 1.53$)이 실제 HPO 정답 피크($\gamma = 1.83$)를 정밀하게 맞추지 못함을 확인(오차 발생). 따라서 ASPIRE 파라미터 튜닝은 무비용 편법 추정 대신, **정석적인 Empirical HPO 탐색을 수행하는 것이 맞다**는 매우 현실적이고 냉정한 결론 도출.
