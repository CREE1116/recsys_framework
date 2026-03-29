# ASPIRE Theory Experiments — 실험 개요

> **ASPIRE**: Adaptive Spectral Power-law Index Restoration for MNAR Bias Correction  
> 스펙트럴 공간에서 인기 편향을 멱법칙으로 모델링하고, β 파라미터로 복원하는 추천 시스템 편향 보정 방법론.

---

## 핵심 이론

**ASPIRE Filter:**
$$h(\sigma_k) = \frac{\sigma_k^{2-2\beta}}{\sigma_k^{2-2\beta} + \alpha}$$

- β=0 → 표준 Tikhonov (MCAR 가정)
- β→1 → 모든 방향 동일 감쇠 (강한 MNAR 보정)
- **β는 Bayesian HPO (Optuna TPE)로 직접 최적화**

**SPP (Spectral Propensity Projection):**
$$\tilde{p}_k = \sum_i V_{ki}^2 \cdot p_i, \quad p_i = n_i / n_{\max}$$

멱법칙 관계: $\log \tilde{p}_k \approx 2\beta \cdot \log \sigma_k + C$

---

## 실험 목록

| 스크립트 | 번호 | 유형 | 핵심 질문 |
|---|---|---|---|
| `exp1_slp.py` | 1 | 이론 검증 | V^T P V 가 대각 우세인가? (SLP 가정 성립) |
| `exp2_power_law.py` | 2 | 이론 검증 | SPP 멱법칙 구조가 실제 데이터에서 나타나는가? |
| `exp3_beta_tracking.py` | 3 | 이론 검증 | MCAR 노이즈 증가 시 β가 0으로 수렴하는가? |
| `exp4_targeted_subsampling.py` | 4 | 이론 검증 | MNAR 강도 η와 β가 단조 관계인가? |
| `exp6_corollary2.py` | 6 | 이론 검증 | β = η/(2α) (Corollary 2) 가 성립하는가? |
| `exp7_beta_sensitivity.py` | 7 | 성능 분석 | β 값에 따라 추천 성능이 어떻게 달라지는가? |
| `exp8_spectral_bias.py` | 8 | 이론 검증 | 상위 방향에 인기 편향이 집중되는가? (SPP 가정) |
| `exp9_filter_effect.py` | 9 | 효과 분석 | 필터 적용 후 아이템 점수가 어떻게 재분배되는가? |
| `exp10_cross_dataset.py` | 10 | 일반화 검증 | Gini, η와 최적 β의 상관이 데이터셋 간에 일관되는가? |
| `exp11_fixed_alpha_beta_sweep.py` | 11 | 시스템 분석 | α 고정 후 β만 스왼 — β의 순수 효과와 α와의 상호작용 분석 |
| `exp12_mcar_hpo_tracking.py` | 12 | 이론 검증 | HPO가 찾는 최적 β도 MNAR 강도에 따라 단조 변화하는가? |
| `exp15_spectral_structure_proof.py` | 15 | 핵심 증명 | SPL이 피드백 루프의 산물임을 시뮬레이션+실제 데이터로 통합 증명 |

---

## 실험별 상세


### Exp 1: SLP Verification
**파일:** `exp1_slp.py`  
**목적:** Spectral Low-rank Popularity (SLP) 가정 검증  
**방법:** M = V^T P V 계산 → ε = mean(|off-diag|) / mean(diag)  
**출력:** `output/slp/{dataset}/` — `slp_heatmap.png`, `result.json`  
**실행:**
```bash
uv run python aspire_experiments/exp1_slp.py --dataset ml1m
```

---

### Exp 2: Power-law Coupling
**파일:** `exp2_power_law.py`  
**목적:** log σ_k vs. log p̃_k 멱법칙 관계 시각화  
**방법:** SPP 계산 → OLS 피팅 → R² 보고  
**출력:** `output/powerlaw/{dataset}/` — `powerlaw_fit.png`, `result.json`  
**실행:**
```bash
uv run python aspire_experiments/exp2_power_law.py --dataset ml1m
```

---

### Exp 3: MCAR Noise Injection → β Tracking
**파일:** `exp3_beta_tracking.py`  
**목적:** 데이터가 MCAR에 가까워질수록 β→0 수렴 검증  
**방법:** 원본 행렬에 균일 랜덤 상호작용 noise_ratio=0~10 주입 → β 추적  
**출력:** `output/tracking/{dataset}/` — `beta_mcar_injection.png`, `result_v3.json`  
**실행:**
```bash
uv run python aspire_experiments/exp3_beta_tracking.py --dataset ml1m ml100k
```

---

### Exp 4: Targeted Subsampling → β Tracking
**파일:** `exp4_targeted_subsampling.py`  
**목적:** MNAR 강도 η와 β의 선형 관계 검증  
**방법:** η 제어 서브샘플링으로 MNAR 강도 조절 → β 추적 (다중 seed)  
**출력:** `output/tracking_v2/{dataset}/` — `beta_vs_eta.png`, `beta_linearity_postcrossover.png`  
**실행:**
```bash
uv run python aspire_experiments/exp4_targeted_subsampling.py --dataset ml1m --seeds 5
```

---

### Exp 6: Corollary 2 Verification
**파일:** `exp6_corollary2.py`  
**목적:** β_theory = η/(2α) ≈ β_measured (SPP+OLS) 검증  
**방법:** 인기도 멱법칙 지수 η, 특이값 지수 α 추정 → 이론 β vs. 측정 β 비교  
**출력:** `output/corollary2/` — `corollary2_scatter.png`, `eta_powerlaw.png`, `alpha_powerlaw.png`, `components_bar.png`  
**실행:**
```bash
uv run python aspire_experiments/exp6_corollary2.py --dataset ml100k ml1m ml20m
```

---

### Exp 7: β Sensitivity Analysis ⭐ 핵심
**파일:** `exp7_beta_sensitivity.py`  
**목적:** β ∈ [0, 0.1, ..., 1.0] sweep → 모든 추천 지표 측정 (사실상 Ablation)  
**방법:** 각 β마다 Bayesian HPO로 최적 α 탐색 → evaluation.yaml 전체 지표 평가  
**출력:** `output/sensitivity/{dataset}/` — 6종 시각화 + `beta_sensitivity.json/csv`  
**실행:**
```bash
uv run python aspire_experiments/exp7_beta_sensitivity.py --dataset ml1m --seeds 42 43 44
```

| 출력 파일 | 내용 |
|---|---|
| `filter_shape_overlay.png` | β별 h(σ) 오버레이 (linear + log-log) |
| `filter_shape_individual.png` | β별 필터 개별 서브플롯 |
| `metric_sweep.png` | β vs. 모든 지표 + α + Head/LT tradeoff |
| `radar_chart.png` | 다차원 성능 레이더 |
| `sensitivity_heatmap.png` | 지표 × β 히트맵 |
| `relative_change.png` | β=0 대비 상대 변화율 (%) |
| `spp_fit.png` | SPP 멱법칙 피팅 (LAD) |

---

### Exp 8: Spectral Bias Analysis
**파일:** `exp8_spectral_bias.py`  
**목적:** SPP 핵심 가정 직접 검증 — "상위 방향에 인기 편향이 집중된다"  
**방법:** 방향별 V²_{ki} 분포, Spearman 상관, 누적 Head 에너지 커브  
**출력:** `output/spectral_bias/{dataset}/` — 4종 시각화  
**실행:**
```bash
uv run python aspire_experiments/exp8_spectral_bias.py --dataset ml1m
```

| 출력 파일 | 내용 |
|---|---|
| `spp_powerlaw.png` | p̃_k vs σ_k 멱법칙, 방향 컬러맵 |
| `head_tail_loading.png` | Head vs. Tail V² 하중 비교 + p̃_k |
| `spearman_per_dir.png` | 방향별 Spearman ρ (하중 ↔ 인기) |
| `cumulative_head_energy.png` | 누적 Head 에너지 편향 커브 |

---

### Exp 9: Filter Effect Analysis
**파일:** `exp9_filter_effect.py`  
**목적:** "ASPIRE 필터가 롱테일 아이템을 실제로 boost하는가" 직접 측정  
**방법:** β=0 vs. β_HPO 아이템 평균 점수 비교, 인기 5분위 분석, Jaccard 중복률  
**출력:** `output/filter_effect/{dataset}/` — 4종 시각화  
**실행:**
```bash
uv run python aspire_experiments/exp9_filter_effect.py --dataset ml1m
```

| 출력 파일 | 내용 |
|---|---|
| `score_delta_vs_rank.png` | Δscore vs. 인기 순위 (인기↑ → 점수↓) |
| `score_delta_boxplot.png` | 인기 5분위별 Δscore 분포 |
| `score_comparison_bar.png` | β=0 vs. β_opt 평균 점수 막대 |
| `jaccard_overlap.png` | 추천 리스트 변화량 @K |

---

### Exp 10: Cross-Dataset β Analysis
**파일:** `exp10_cross_dataset.py`  
**목적:** "불균등한 데이터일수록 더 큰 β가 필요" 가설 검증  
**방법:** 데이터셋별 Gini, η 계산 → HPO 최적 β와 상관 분석 + β_OLS vs β_HPO 비교  
**출력:** `output/cross_dataset/` — 4종 시각화 + `summary.csv`  
**실행:**
```bash
uv run python aspire_experiments/exp10_cross_dataset.py --datasets ml100k ml1m ml20m
```

| 출력 파일 | 내용 |
|---|---|
| `gini_vs_beta.png` | Gini ↔ β_HPO 산점도 |
| `eta_vs_beta.png` | η ↔ β_HPO 산점도 |
| `ols_vs_hpo_beta.png` | β_OLS vs. β_HPO (추정 정확도) |
| `dataset_heatmap.png` | 데이터셋 특성 비교 히트맵 |

---

### Exp 11: Fixed-α β Sweep
**파일:** `exp11_fixed_alpha_beta_sweep.py`  
**목적:** α 고정 후 β만 스왼 — β의 **순수 효과**와 α와의 **상호작용** 분석  
**방법:** 3가지 α 기준(β=0 최적, OLS β 기반, joint HPO)을 고정 후 β ∈ [0,0.1,...,1.0] sweep  
**출력:** `output/fixed_alpha/{dataset}/` — 4종 시각화  
**실행:**
```bash
uv run python aspire_experiments/exp11_fixed_alpha_beta_sweep.py --dataset ml1m
```

| 출력 파일 | 내용 |
|---|---|
| `beta_sweep_3panel.png` | NDCG / LT-NDCG / Coverage × β (3개 α 비교) |
| `head_lt_tradeoff.png` | Head vs. LT NDCG 트레이드오프 |
| `filter_shapes.png` | α별 h(σ) 필터 형태 (linear + log-log) |
| `alpha_beta_heatmap.png` | α 기준 × β 히트맵 (NDCG@10) |

---

### Exp 12: MCAR Noise + HPO β Tracking
**파일:** `exp12_mcar_hpo_tracking.py`  
**목적:** β가 HPO로 최적화되는 세계에서도 **HPO가 알는 최적 β가 MNAR 강도를 정확히 추적하는지** 직접 검증  
**방법:** MCAR 노이즈 주입(noise_ratio=0~4) 후 joint HPO로 (α*, β*) 탐색 → β*가 MCAR 버전에 따라 0으로 수렴하는지 확인  
**출력:** `output/mcar_hpo/{dataset}/` — 4종 시각화  
**실행:**
```bash
uv run python aspire_experiments/exp12_mcar_hpo_tracking.py --dataset ml100k
```

| 출력 파일 | 내용 |
|---|---|
| `beta_tracking.png` | MCAR fraction vs. β_OLS + β_HPO 비교 |
| `performance_tracking.png` | MCAR fraction vs. NDCG / LT-NDCG / Coverage |
| `ols_vs_hpo.png` | β_OLS vs. β_HPO 산점도 (y=x 일치성) |
| `filter_per_noise.png` | MCAR 수준별 필터 h(σ) 형태 |

---

### Exp 15: Spectral Structure Proof (Simulation + Yahoo R3)
**파일:** `exp15_spectral_structure_proof.py`  
**목적:** "스펙트럴 멱법칙(SPL)은 추천 시스템의 피드백 루프가 만들어낸 구조적 산물이다"라는 명제를 증명  
**방법:**
1.  **Simulation**: 무작위 데이터에서 시작해 10회 루프 후 SPL 구조($R^2 \uparrow$) 출현 확인
2.  **Real Data**: Yahoo R3의 MNAR(편향) vs MCAR(비편향) 스펙트럼 직접 비교
**출력:** `output/exp15/` — `spl_proof_master.png` (2패널), `results.json`  
**실행:**
```bash
uv run python aspire_experiments/exp15_spectral_structure_proof.py --iter 10 --temp 0.1
```

---

## 한번에 전체 실행

```bash
# 전체 실험 (기본 설정)
uv run python aspire_experiments/run_aspire_theory_experiments.py \
    --datasets ml100k ml1m \
    --seeds 42 43 44

# 무거운 실험 제외 (이론 검증만)
uv run python aspire_experiments/run_aspire_theory_experiments.py \
    --datasets ml100k ml1m \
    --skip exp7 exp9 exp10 exp11

# 특정 실험만 (exp7만)
uv run python aspire_experiments/run_aspire_theory_experiments.py \
    --datasets ml100k ml1m \
    --skip exp1 exp2 exp3 exp4 exp6 exp8 exp9 exp10 exp11
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--datasets` | `ml1m` | 데이터셋 목록 |
| `--energy` | `0.95` | SVD target energy |
| `--seeds` | `42` | 랜덤 시드 목록 |
| `--n_trials` | `60` | Bayesian HPO trial 수 |
| `--patience` | `20` | HPO 조기 종료 patience |
| `--skip` | `[]` | 건너뛸 실험 (exp1~exp12) |

---

## 결과 구조

```
aspire_experiments/output/
├── slp/               # Exp 1
├── powerlaw/          # Exp 2
├── tracking/          # Exp 3
├── tracking_v2/       # Exp 4
├── corollary2/        # Exp 6
├── sensitivity/       # Exp 7 ⭐
├── spectral_bias/     # Exp 8
├── filter_effect/     # Exp 9
├── cross_dataset/     # Exp 10
├── fixed_alpha/       # Exp 11
├── mcar_hpo/          # Exp 12
└── summary/           # 마스터 요약 CSV
```

---

## 공통 유틸리티 (`exp_utils.py`)

| 함수 | 설명 |
|---|---|
| `load_config(dataset_name)` | YAML 데이터셋 설정 로드 |
| `get_loader_and_svd(...)` | DataLoader + SVD 초기화 |
| `get_eval_config(loader)` | `evaluation.yaml` 기준 평가 설정 로드 |
| `ensure_dir(path)` | 디렉토리 생성 |
| `AspireHPO(...)` | Optuna TPE 기반 Bayesian HPO 래퍼 |
