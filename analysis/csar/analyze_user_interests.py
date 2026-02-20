"""
CSAR 유저 관심사 분석 스크립트

분석 항목:
1. 관심사별 유저 인구통계 분석 (성별, 연령, 직업)
2. Top-N 유저 분석 (관심사별 가장 활성화된 유저)
3. 카이제곱 검정 (관심사와 인구통계 간 연관성)
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import json
import yaml


def load_model(exp_path: str, device='cpu'):
    """학습된 모델 로드"""
    from src.models import MODEL_REGISTRY
    from src.data_loader import DataLoader
    
    config_path = Path(exp_path) / 'config.yaml'
    model_path = Path(exp_path) / 'best_model.pt'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = DataLoader(config)
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class(config, data_loader)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, data_loader, config


def load_user_metadata(data_dir):
    """ML-1M 유저 메타데이터 로드"""
    if os.path.isfile(data_dir):
        data_dir = os.path.dirname(data_dir)
    
    users_path = os.path.join(data_dir, 'users.dat')
    if not os.path.exists(users_path):
        print(f"[Error] users.dat not found in {data_dir}")
        return None
    
    # ML-1M 포맷: UserID::Gender::Age::Occupation::Zip-code
    users = pd.read_csv(users_path, sep='::', engine='python', header=None,
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
    
    # 연령대 매핑
    age_map = {
        1: "Under 18", 18: "18-24", 25: "25-34",
        35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
    }
    users['age_group'] = users['age'].map(age_map)
    
    # 직업 매핑
    occupation_map = {
        0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
        4: "college/grad student", 5: "customer service", 6: "doctor/health care",
        7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
        11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
        15: "scientist", 16: "self-employed", 17: "technician/engineer",
        18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
    }
    users['occupation_name'] = users['occupation'].map(occupation_map)
    
    return users.set_index('user_id')


def get_user_memberships(model, data_loader, device):
    """모든 유저의 멤버십 계산"""
    model.eval()
    with torch.no_grad():
        user_ids = torch.arange(data_loader.n_users).to(device)
        user_embs = model.user_embedding(user_ids)
        memberships = model.attention_layer.get_membership(user_embs).cpu().numpy()
    return memberships


def analyze_interest_demographics(memberships, users_df, top_n=100):
    """관심사별 인구통계 분석"""
    K = memberships.shape[1]
    results = []
    
    for k in range(K):
        # Top-N 유저 선택
        top_indices = np.argsort(memberships[:, k])[-top_n:]
        top_user_ids = top_indices + 1  # ML-1M은 1-indexed
        
        # 메타데이터 매칭
        matched = users_df.loc[users_df.index.isin(top_user_ids)]
        
        if len(matched) == 0:
            continue
        
        # 성별 분포
        gender_dist = matched['gender'].value_counts(normalize=True).to_dict()
        male_ratio = gender_dist.get('M', 0)
        
        # 연령대 분포
        age_dist = matched['age_group'].value_counts(normalize=True).to_dict()
        top_age = matched['age_group'].mode().iloc[0] if len(matched) > 0 else "N/A"
        
        # 직업 분포
        occ_dist = matched['occupation_name'].value_counts(normalize=True).to_dict()
        top_occ = matched['occupation_name'].mode().iloc[0] if len(matched) > 0 else "N/A"
        
        results.append({
            'interest': k,
            'n_users': len(matched),
            'male_ratio': male_ratio,
            'female_ratio': 1 - male_ratio,
            'top_age': top_age,
            'age_dist': age_dist,
            'top_occupation': top_occ,
            'occ_dist': occ_dist,
        })
    
    return results


def chi_square_test(memberships, users_df, attribute='gender', top_n=100):
    """관심사와 인구통계 속성 간 카이제곱 검정"""
    K = memberships.shape[1]
    
    # 각 유저의 주요 관심사 결정
    primary_interests = memberships.argmax(axis=1)
    
    # 속성별로 Contingency Table 생성
    data = []
    for user_idx in range(len(primary_interests)):
        user_id = user_idx + 1
        if user_id in users_df.index:
            data.append({
                'interest': primary_interests[user_idx],
                'attribute': users_df.loc[user_id, attribute]
            })
    
    df = pd.DataFrame(data)
    contingency = pd.crosstab(df['interest'], df['attribute'])
    
    # 카이제곱 검정
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': contingency,
        'significant': p_value < 0.05
    }


def visualize_demographics(results, save_path):
    """인구통계 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df = pd.DataFrame(results)
    
    # 1. 성별 비율 by Interest
    axes[0, 0].bar(df['interest'], df['male_ratio'], label='Male', alpha=0.7)
    axes[0, 0].bar(df['interest'], df['female_ratio'], bottom=df['male_ratio'], label='Female', alpha=0.7)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[0, 0].set_xlabel('Interest')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].set_title('Gender Distribution by Interest')
    axes[0, 0].legend()
    
    # 2. 성별 편향 (Male Ratio)
    colors = ['blue' if r > 0.5 else 'red' for r in df['male_ratio']]
    axes[0, 1].bar(df['interest'], df['male_ratio'] - 0.5, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Interest')
    axes[0, 1].set_ylabel('Male Bias (ratio - 0.5)')
    axes[0, 1].set_title('Gender Bias by Interest')
    
    # 3. Top Age Group
    age_counts = df['top_age'].value_counts()
    axes[1, 0].pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Most Common Age Group per Interest')
    
    # 4. Top Occupation
    occ_counts = df['top_occupation'].value_counts().head(10)
    axes[1, 1].barh(occ_counts.index, occ_counts.values, alpha=0.7)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Top 10 Occupations Across Interests')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(results, chi_results, output_path):
    """마크다운 리포트 생성"""
    lines = ["# User Interest Demographics Analysis\n\n"]
    
    # 요약
    lines.append("## Summary\n\n")
    lines.append(f"- **Number of Interests**: {len(results)}\n")
    
    # 카이제곱 결과
    lines.append("\n## Chi-Square Test Results\n\n")
    for attr, res in chi_results.items():
        sig = "✅ Significant" if res['significant'] else "❌ Not Significant"
        lines.append(f"### {attr.title()}\n")
        lines.append(f"- χ² = {res['chi2']:.2f}\n")
        lines.append(f"- p-value = {res['p_value']:.4e}\n")
        lines.append(f"- Result: {sig}\n\n")
    
    # 관심사별 분석
    lines.append("## Interest-Level Demographics\n\n")
    lines.append("| Interest | Male% | Top Age | Top Occupation |\n")
    lines.append("|----------|-------|---------|----------------|\n")
    
    for r in sorted(results, key=lambda x: x['interest']):
        lines.append(f"| {r['interest']} | {r['male_ratio']*100:.1f}% | {r['top_age']} | {r['top_occupation']} |\n")
    
    with open(output_path, 'w') as f:
        f.writelines(lines)


# ============================================================
# 기본 설정 (수정 가능)
# ============================================================
EXP_PATH = 'trained_model/ml-1m-10c/csar-rec2'
DEVICE = 'cpu'
TOP_N = 100


def main():
    exp_path = EXP_PATH
    device = DEVICE
    top_n = TOP_N
    
    print(f"Loading model from {exp_path}...")
    model, data_loader, config = load_model(exp_path, device)
    
    # 유저 메타데이터 로드
    data_dir = os.path.dirname(config['data_path'])
    users_df = load_user_metadata(data_dir)
    if users_df is None:
        return
    
    print(f"Loaded {len(users_df)} users with metadata")
    
    # 멤버십 계산
    print("Computing user memberships...")
    memberships = get_user_memberships(model, data_loader, device)
    
    print("\n" + "=" * 60)
    print("User Interest Demographics Analysis")
    print("=" * 60)
    
    # 관심사별 인구통계
    print(f"\n[1] Analyzing Top-{top_n} Users per Interest...")
    results = analyze_interest_demographics(memberships, users_df, top_n)
    
    # 카이제곱 검정
    print("\n[2] Chi-Square Tests...")
    chi_results = {}
    for attr in ['gender', 'age_group', 'occupation_name']:
        print(f"  - {attr}...", end=' ')
        chi_res = chi_square_test(memberships, users_df, attr, top_n)
        chi_results[attr] = chi_res
        sig = "✅" if chi_res['significant'] else "❌"
        print(f"χ²={chi_res['chi2']:.1f}, p={chi_res['p_value']:.2e} {sig}")
    
    # 출력 경로 설정 (output 폴더)
    from analysis.utils import get_analysis_output_path
    run_name = os.path.basename(exp_path)
    output_dir = Path(get_analysis_output_path(config['dataset_name'], run_name))
    
    # 시각화
    print("\n[3] Visualization...")
    fig_path = output_dir / 'user_demographics_analysis.png'
    visualize_demographics(results, fig_path)
    print(f"Saved: {fig_path}")
    
    # 리포트 생성
    report_path = output_dir / 'user_demographics_report.md'
    generate_report(results, chi_results, report_path)
    print(f"Saved: {report_path}")
    
    # JSON 저장
    stats_path = output_dir / 'user_demographics_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'results': [
                {k: v for k, v in r.items() if k not in ['age_dist', 'occ_dist']} 
                for r in results
            ],
            'chi_square': {k: {
                'chi2': float(v['chi2']), 
                'p_value': float(v['p_value']), 
                'significant': bool(v['significant'])
            } for k, v in chi_results.items()}
        }, f, indent=2)
    print(f"Saved: {stats_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

