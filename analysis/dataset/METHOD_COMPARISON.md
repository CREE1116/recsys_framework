# Trade-off 분석 방법론 비교

## 📋 개요

다양성-정확도 Trade-off를 정량화하는 3가지 방법론의 **차이점**과 **계산 방식**을 정리합니다.

---

## 🔬 방법론 1: 확률 질량 이동 (Probability Mass Shift)

### 핵심 아이디어

> "정답이 몰려 있는 Head에서 Tail로 추천을 옮기면, 정확도가 떨어지는 것은 수학적 필연"

### 계산 방식

```
1. δ (Delta) = LTC_model - LTC_baseline
   → Tail로 이동한 추천 비중

2. Natural Decay Factor = P_head - P_tail
   → Head와 Tail 아이템의 정답 확률 차이 (데이터에서 측정)

3. Expected Loss = δ × Decay × Baseline_NDCG
   → 이론적 예상 손실

4. Actual Loss = Baseline_NDCG - Model_NDCG
   → 실제 손실

5. Outperformance = (Actual - Expected) / Expected × 100%
   → 이론 대비 초과 성능 (양수 = 이론보다 좋음)
```

### 특징

| 항목            | 설명                                  |
| --------------- | ------------------------------------- |
| **관점**        | 확률론적/베이지안                     |
| **비교 대상**   | Baseline 모델 (예: MF)                |
| **장점**        | 직관적, 계산 간단                     |
| **한계**        | 선형 근사, Baseline 선택에 민감       |
| **적합한 상황** | "X% Tail 증가 시 Y% 정확도 손실" 설명 |

---

## 📐 방법론 2: 파레토 효율성 지수 (PEI)

### 핵심 아이디어

> "지능 없이 섞은 기준선(Linear Interpolation)보다 얼마나 '위'에 있는가?"

### 계산 방식

```
1. Oracle Points 식별
   - Accuracy Oracle (A): NDCG 최대, LTC 최소 (예: Most-Popular)
   - Diversity Oracle (B): LTC 최대, NDCG 최소 (예: Random)

2. Linear Interpolation (기준선)
   Expected_NDCG(x) = NDCG_A + slope × (x - LTC_A)
   where slope = (NDCG_B - NDCG_A) / (LTC_B - LTC_A)

3. PEI = NDCG_actual / NDCG_expected
   → 1 = 기준선과 동일
   → >1 = 효율적 Trade-off
   → >>1 = 매우 효율적 (파레토 프론티어)
```

### 시각적 이해

```
NDCG ▲
     │  ● A (Most-Popular)
     │     \
     │      \  Linear Baseline
     │    ★  \   ← 모델이 여기 있으면 PEI > 1
     │        \
     │         ● B (Random)
     └──────────────────► LTC
```

### 특징

| 항목            | 설명                                     |
| --------------- | ---------------------------------------- |
| **관점**        | 경제학적 (파레토 최적)                   |
| **비교 대상**   | 두 극단 Oracle 사이 선형 보간선          |
| **장점**        | 다양성 수준이 다른 모델도 공정 비교 가능 |
| **한계**        | Oracle 선택에 의존, 선형 가정            |
| **적합한 상황** | "기준 대비 N배 효율적" 주장              |

---

## 🧮 방법론 3: KL-Divergence Bound

### 핵심 아이디어

> "다양성 증가 = 엔트로피 증가 = 데이터 분포와의 거리(KL) 증가 = 정확도 하락"

### 계산 방식

```
1. Data Entropy: H(P_data)
   → 훈련 데이터의 아이템 분포 엔트로피 (Power-law → 낮음)

2. Uniform Entropy: H(Uniform) = log(N_items)
   → 완전 균등 분포의 엔트로피 (최대값)

3. Recommendation Entropy: H(P_rec)
   → 추천 분포의 엔트로피 (모델에서 측정)

4. KL-Divergence:
   D_KL(P_rec || P_data)
   → 추천 분포가 데이터 분포에서 얼마나 벗어났는지

5. Lower Bound:
   ↑ H(P_rec) → ↑ D_KL → ↓ Accuracy
```

### 정보 이론적 직관

```
P_data:    [0.5, 0.3, 0.1, 0.05, ...]  ← 뾰족함 (저엔트로피)
P_uniform: [0.1, 0.1, 0.1, 0.1, ...]   ← 평평함 (고엔트로피)

다양성 추구 = P_rec를 Uniform에 가깝게 만들기
            = P_data에서 멀어지기
            = KL-Divergence 증가
            = 정확도 손실
```

### 특징

| 항목            | 설명                                             |
| --------------- | ------------------------------------------------ |
| **관점**        | 정보 이론                                        |
| **비교 대상**   | 데이터 분포 P_data                               |
| **장점**        | 수학적으로 엄밀, 학술적 신뢰성                   |
| **한계**        | 직관적 해석 어려움, 정확한 Lower Bound 도출 복잡 |
| **적합한 상황** | 이론적 정당화, Discussion 섹션                   |

---

## ⚖️ 세 방법론 비교 요약

| 측면            | Mass Shift         | PEI                  | KL-Divergence         |
| --------------- | ------------------ | -------------------- | --------------------- |
| **핵심 질문**   | "얼마나 손해봤나?" | "얼마나 효율적인가?" | "왜 손해가 필연인가?" |
| **수식 복잡도** | ⭐ 낮음            | ⭐⭐ 중간            | ⭐⭐⭐ 높음           |
| **계산 입력**   | Baseline 모델      | Oracle 2개           | 전체 분포             |
| **출력 해석**   | Outperformance %   | PEI (배수)           | 엔트로피 Gap          |
| **논문 위치**   | Experiments        | Analysis             | Discussion            |
| **시각화**      | 손실 비교 막대     | Trade-off 곡선       | Entropy 산점도        |

---

## 🎯 언제 어떤 방법을 쓸까?

### 1. 실험 결과 섹션 (Main Results)

→ **Mass Shift** 사용

```
"CSAR는 Tail 12.9% 달성 시 이론적으로 5% 손실이 예상되었으나,
 실제 손실은 2%에 불과했다."
```

### 2. 분석 섹션 (Further Analysis)

→ **PEI** 사용

```
"CSAR의 PEI는 3.79로, 단순 혼합 대비 279% 효율적인
 파레토 프론티어에 위치한다."
```

### 3. 이론적 배경/Discussion

→ **KL-Divergence** 언급

```
"정보 이론적으로, 추천 엔트로피 증가는 P_data와의
 KL-Divergence 증가를 수반하며, 이는 정확도 하락의
 수학적 하한을 형성한다."
```

---

## 📊 실제 계산 예시 (ML-100k)

### 데이터셋 통계

- Items: 1,008 (Head: 202, Tail: 806)
- Head Popularity: 58.4%
- Tail Popularity: 41.6%
- Natural Decay Factor: 0.168

### CSAR-Rec2 (K=64) 결과

| 방법론         | 계산                                              | 결과                   |
| -------------- | ------------------------------------------------- | ---------------------- |
| **Mass Shift** | (0.125 - 0.02) × 0.168 × 0.053 = 0.0009 예상 손실 | Outperformance: +30.4% |
| **PEI**        | 0.068 / 0.018 = 3.79                              | 기준 대비 279% 효율    |
| **KL**         | H(rec) = 9.2, H(data) = 6.4, Gap = 2.8            | 엔트로피 갭 상당       |

---

## 🔧 스크립트 사용법

```bash
# 기본 (NDCG vs LongTailCoverage)
uv run python analysis/dataset/tradeoff_theoretical_analysis.py trained_model/ml-100k

# 지표 선택 (LongTailNDCG vs Entropy)
uv run python analysis/dataset/tradeoff_theoretical_analysis.py trained_model/ml-100k \
    --accuracy LongTailNDCG --diversity Entropy

# 옵션
--accuracy: NDCG, HitRate, LongTailNDCG, LongTailHitRate, HeadNDCG, HeadHitRate
--diversity: LongTailCoverage, Entropy, Coverage, ILD, Novelty
```
