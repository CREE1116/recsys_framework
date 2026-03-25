# ASPIRE Experiments: Methodology Summary (Refined v3)

... (Previous sections) ...

## 실험 8: Gamma-Alpha Ablation Study (구성요소 기여도 분석)
- **방법**: 이론값(Theory)과 최적값(HPO)을 조합한 4가지 모드(V1~V4)를 비교하고, NDCG뿐만 아니라 **Coverage**와 **Tail-NDCG**를 함께 측정함.
- **결과**: 이론적 $\gamma$는 높은 Coverage와 롱테일 성능을 보장하지만, 최고 정확도(Overall NDCG)를 위해서는 $\alpha$와 $\gamma$의 공동 최적화가 필수적임을 확인.

## 실험 9: Beta-Gamma Stability Analysis (상위 성분 제외 분석)
- **방법**: 최상위 특이값($\sigma_1 \sim \sigma_5$)을 복원 연산에서 차례로 제외(Skip)하며 HPO를 수행하여, 'Goliath' 성분이 최적 파라미터 추정에 미치는 영향을 분석함.
- **결과**: 상위 2~3개 성분만 제외해도 최적 $\gamma$가 이론적 예측치(OLS/LAD 모두 약 1.1)와 거의 완벽하게 일치하며 안정화됨을 확인. 이는 최상위 대역의 전역적 편향이 실제 데이터의 Latent Power-law 지수 추정을 방해하고 있었음을 시사함.
