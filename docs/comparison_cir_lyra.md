# CIR vs. LYRA 모델 비교 분석

최신 추천시스템 프레임워크 내에서 구현된 두 Closed-form 모델인 **CIR(Closed-form Item Resonance)**과 **LYRA(Ridge Regression)**의 기술적 차이점을 정리합니다.

## 1. 개요 (Abstract)

- **CIR**: 아이템 간의 상관관계(Correlation)와 공명(Resonance)에 집중하며, 아이템 기반의 필터링 관점에서 접근합니다.
- **LYRA-Minimal**: 모든 하이퍼파라미터를 걷어내고, SVD를 통한 정보 압축과 Varimax를 통한 관심사 정렬(Interest Alignment) 본연의 기능에만 집중합니다.

## 2. 수식 및 알고리즘 비교

| 항목          | CIR (Current)                | LYRA-Minimal (Resonance)                 |
| :------------ | :--------------------------- | :--------------------------------------- |
| **핵심 철학** | Correlation Filter           | **Interest Resonance**                   |
| **기저 정렬** | -                            | **Varimax Rotation**                     |
| **전이 학습** | $S = (C + \lambda I)^{-1} C$ | **$I - G_{inv} / \text{diag}(G_{inv})$** |
| **복잡도**    | Medium                       | **Minimal (Low)**                        |
| **파라미터**  | $\lambda$ (Regularization)   | **K** (Latent Dimension)                 |
| **추론 방식** | Inductive                    | Inductive (Implicitly)                   |

## 3. 핵심 차이점 상세

- **LYRA-ALR**: $\log(1 + X^TX)$를 통해 인기 아이템의 지배력을 억제하고 테일 신호를 증폭합니다.
- **CIR**: 포스트 정규화를 통해 에너지 밸런스를 맞춥니다.

### 3.2 수축 및 공명 매커니즘

- **ALR-Equalized**: EASE 스타일의 Diagonal Zero 제약을 잠재 공간에 적용합니다. 특히 화이트닝 기저($V \cdot L^{-0.5}$)를 사용하여 모든 취향 축이 동일한 발언권을 갖도록 설계되었습니다.
- **CIR**: 상관관계 행렬의 에너지를 직접 수축(Shrinkage)하여 필터링합니다.

## 4. 결론

**LYRA-ALR (Equalized)**는 표준적인 릿지 회귀를 넘어, 취향의 다양성을 보존하고 롱테일 아이템의 전파력을 극대화한 하이엔드 닫힌 해 모델입니다.
