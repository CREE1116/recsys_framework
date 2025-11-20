import os
import re
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency, pearsonr

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import load_model_from_run, get_analysis_output_path, AnalysisReport


# --------------------------------------------------------------------------------
# 1. 데이터 로드 함수 (증강된 메타데이터 로드용)
# --------------------------------------------------------------------------------
def load_augmented_metadata(data_dir):
    """ 크롤링된 movies_augmented.csv 파일을 로드합니다. (경로 에러 수정판) """
    
    # [핵심 수정] data_dir가 파일(ratings.dat)을 가리키면 부모 폴더 경로로 보정
    if os.path.isfile(data_dir):
        data_dir = os.path.dirname(data_dir)

    path = os.path.join(data_dir, 'movies_augmented.csv')
    
    if not os.path.exists(path):
        print(f"[Warning] Augmented metadata not found at {path}. Using basic metadata.")
        # 없으면 기본 movies.dat 로드 (기존 로직 fallback)
        # 인코딩 에러 방지를 위해 latin-1 사용
        return pd.read_csv(os.path.join(data_dir, 'movies.dat'), sep='::', engine='python', 
                           names=['item_id', 'title', 'genres'], encoding='latin-1').set_index('item_id')
    
    df = pd.read_csv(path)
    # 전처리: NaN 채우기
    df = df.fillna('')
    return df.set_index('item_id')

# --------------------------------------------------------------------------------
# 2. 핵심 분석 함수 (장르/연도 + 감독/배우/키워드 추가)
# --------------------------------------------------------------------------------
def generate_deep_interest_report(report, interest_k, item_interests, inv_item_map, metadata_df, global_stats, top_n=20, interest_key_norm=None):
    """
    모든 메타데이터(장르, 연도, 감독, 배우, 키워드)에 대해 과잉 표집 분석을 수행합니다.
    """
    report.add_section(f"Interest #{interest_k} Deep Analysis", level=3)
    
    stats_summary = {'k': interest_k}

    # [1] Norm & Popularity
    if interest_key_norm is not None:
        norm_val = interest_key_norm.item() if isinstance(interest_key_norm, torch.Tensor) else interest_key_norm
        report.add_text(f"- **Key Norm:** `{norm_val:.4f}`")
        stats_summary['norm'] = norm_val

    # [2] Top-N Items Extraction
    weights_for_k = item_interests[:, interest_k]
    top_weights, top_indices = torch.topk(weights_for_k, top_n)
    
    top_items_info = []
    top_item_pops = []
    
    # Top-N 아이템들의 메타데이터 수집
    current_features = {
        'genres': [], 'decades': [], 'directors': [], 'cast': [], 'keywords': []
    }

    for i, model_item_id in enumerate(top_indices.cpu().numpy()):
        original_item_id = inv_item_map.get(model_item_id)
        pop_count = global_stats['item_counts'].get(model_item_id, 0)
        top_item_pops.append(pop_count)
        
        if original_item_id and original_item_id in metadata_df.index:
            row = metadata_df.loc[original_item_id]
            
            # 제목 정제
            title = row['title']
            title_cleaned = re.sub(r'\s*\(\d{4}\)', '', title).strip()
            
            # 연도/연대 처리
            year = None
            match = re.search(r'\((\d{4})\)', title)
            if match:
                year = int(match.group(1))
                current_features['decades'].append(int(np.floor(year / 10) * 10))

            # 장르 처리
            genres = row['genres'].split('|') if row['genres'] else []
            current_features['genres'].extend(genres)

            # [추가] 감독 처리
            director = row.get('director', '')
            if director: current_features['directors'].append(director)

            # [추가] 배우 처리
            cast = row.get('cast', '').split('|') if row.get('cast') else []
            current_features['cast'].extend(cast)

            # [추가] 키워드 처리
            kws = row.get('keywords', '').split('|') if row.get('keywords') else []
            current_features['keywords'].extend(kws)

            top_items_info.append({
                'Rank': i + 1, 'Title': title_cleaned, 'Year': year, 'Genres': ', '.join(genres[:2]),
                'Director': director, 'Weight': f"{top_weights[i].item():.4f}"
            })

    # 테이블 출력
    if top_items_info:
        df_show = pd.DataFrame(top_items_info)
        report.add_text(f"**Top {top_n} Items**")
        report.add_table(df_show[['Rank', 'Title', 'Year', 'Genres', 'Director', 'Weight']]) # 감독 컬럼 추가

    # 인기도 요약
    if top_item_pops:
        avg_log_pop = np.mean(np.log1p(top_item_pops))
        stats_summary['avg_popularity'] = avg_log_pop

    # ---------------------------------------------------------
    # [공통] 통계적 유의성 검정 함수 (재사용)
    # ---------------------------------------------------------
    def run_test(feature_list, global_counter, feature_name, min_count=2):
        """
        feature_list: Top-N 아이템들의 특성 리스트 (예: ['Action', 'Drama', ...])
        global_counter: 전체 데이터셋의 특성 카운터
        """
        target_counts = Counter(feature_list)
        results = []
        
        # Top-N 내에서 최소 n번 이상 등장한 특성만 검사 (노이즈 제거)
        candidates = [k for k, v in target_counts.items() if v >= min_count]
        
        for feat in candidates:
            count_in_top = target_counts[feat]
            count_not_in_top = len(feature_list) - count_in_top # 해당 feature list 길이 기준
            
            global_count = global_counter.get(feat, 0)
            # 전체 데이터셋에서의 feature list 총 길이 (근사값)
            total_feature_count = sum(global_counter.values())
            
            count_in_others = global_count - count_in_top
            count_not_in_others = (total_feature_count - len(feature_list)) - count_in_others
            
            # 2x2 행렬
            table = [[count_in_top, count_not_in_top], [count_in_others, count_not_in_others]]
            
            try:
                chi2, p_value, _, _ = chi2_contingency(table, correction=True)
                
                top_ratio = count_in_top / len(feature_list)
                global_ratio = global_count / total_feature_count
                
                if p_value < 0.05 and top_ratio > global_ratio:
                    lift = top_ratio / global_ratio if global_ratio > 0 else 0
                    results.append({
                        'Feature': feat, 'Count': count_in_top, 
                        'Lift': f"{lift:.1f}x", 'p-value': f"{p_value:.4f}"
                    })
            except: pass # 0 division 등 방지
            
        if results:
            df = pd.DataFrame(results).sort_values('p-value', ascending=True).head(5) # Top 5만 표시
            report.add_text(f"\n**Significant {feature_name} (p<0.05)**")
            report.add_table(df)
            return df.iloc[0]['Feature'] # 가장 강력한 특징 리턴
        return None

    # ---------------------------------------------------------
    # 3. 분야별 상세 분석 실행
    # ---------------------------------------------------------
    
    # (1) 장르 & 연대
    top_genre = run_test(current_features['genres'], global_stats['genres'], "Genres")
    top_decade = run_test(current_features['decades'], global_stats['decades'], "Decades")
    
    # (2) 감독 (Director) - 핵심!
    top_director = run_test(current_features['directors'], global_stats['directors'], "Directors", min_count=2)
    
    # (3) 배우 (Cast)
    top_cast = run_test(current_features['cast'], global_stats['cast'], "Cast", min_count=3)
    
    # (4) 키워드 (Keywords)
    top_keyword = run_test(current_features['keywords'], global_stats['keywords'], "Keywords", min_count=3)

    # ---------------------------------------------------------
    # 4. 정성적 요약 (자동 생성)
    # ---------------------------------------------------------
    report.add_section("Qualitative Summary", level=4)
    summary = f"Interest #{interest_k} shows specific preferences."
    
    points = []
    if top_director: points.append(f"films by **{top_director}**")
    if top_genre: points.append(f"**{top_genre}** genre")
    if top_decade: points.append(f"from the **{top_decade}s**")
    if top_keyword: points.append(f"related to **'{top_keyword}'**")
    
    if points:
        summary = f"Interest #{interest_k} captures **{', '.join(points)}**."
    
    report.add_text(summary)
    return stats_summary

# --------------------------------------------------------------------------------
# 3. 메인 실행 함수
# --------------------------------------------------------------------------------
def run_full_analysis(exp_config):
    run_folder_path = exp_config['run_folder_path']
    top_n = exp_config.get('top_n', 20)
    
    print(f"Running Deep Semantic Analysis for {run_folder_path}...")
    
    # 모델 로드
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return

    # 데이터 로드 (증강된 데이터)
    metadata_df = load_augmented_metadata(config['data_path'])
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}
    
    # 전역 통계 계산 (비교군 생성을 위해 미리 한 번 훑기)
    print("Calculating global statistics...")
    global_stats = {
        'item_counts': data_loader.df['item_id'].value_counts().to_dict(),
        'genres': Counter(), 'decades': Counter(), 
        'directors': Counter(), 'cast': Counter(), 'keywords': Counter()
    }
    
    for idx, row in metadata_df.iterrows():
        # Genre
        if row['genres']: global_stats['genres'].update(row['genres'].split('|'))
        # Decade
        match = re.search(r'\((\d{4})\)', str(row['title']))
        if match: global_stats['decades'].update([int(np.floor(int(match.group(1))/10)*10)])
        # Director
        if row.get('director'): global_stats['directors'].update([row['director']])
        # Cast
        if row.get('cast'): global_stats['cast'].update(str(row['cast']).split('|'))
        # Keywords
        if row.get('keywords'): global_stats['keywords'].update(str(row['keywords']).split('|'))

    # 리포트 생성
    run_name = os.path.basename(run_folder_path)
    output_path = get_analysis_output_path(config['dataset_name'], run_name)
    report = AnalysisReport(f"Deep Semantic Analysis: {run_name}", output_path)
    
    # 분석 루프
    with torch.no_grad():
        item_interests = model.attention_layer(model.item_embedding.weight)
        keys_norm = torch.norm(model.attention_layer.interest_keys, p=2, dim=1).cpu()
    
    collected_stats = []
    for k in range(model.num_interests):
        stats = generate_deep_interest_report(
            report, k, item_interests, inv_item_map, metadata_df, global_stats, top_n, keys_norm[k]
        )
        if stats: collected_stats.append(stats)
        
    # 글로벌 상관관계 그래프 (이전 코드와 동일)
    if collected_stats:
        report.add_section("3. Global Analysis: Adaptive Scaling Verification", level=2)
        
        df_stats = pd.DataFrame(collected_stats)
        
        # 상관계수 계산
        corr, p_val = pearsonr(df_stats['avg_popularity'], df_stats['norm'])
        
        report.add_text(f"- **Pearson Correlation (Norm vs Log-Pop):** `{corr:.4f}`")
        report.add_text(f"- **P-value:** `{p_val:.4e}`")
        
        # 해석 자동 생성
        if corr < -0.3:
            interpretation = "✅ **Result: Strong Negative Correlation.**\n\nThis confirms the **Adaptive Scaling Hypothesis**. The model automatically reduces the norm of interest keys associated with popular items (high magnitude embeddings) to maintain non-linearity in the Softplus activation function."
        elif corr > 0.3:
            interpretation = "⚠️ **Result: Positive Correlation.**\nThe model assigns larger norms to popular interests, potentially amplifying popularity bias."
        else:
            interpretation = "⏺️ **Result: No Significant Correlation.**\nThe model maintains consistent norms across different popularity levels, suggesting a robust representation independent of item frequency."
        
        report.add_text(interpretation)
        
        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df_stats, x='avg_popularity', y='norm', scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red'})
        
        # 각 점에 ID 표시
        for i, row in df_stats.iterrows():
            plt.text(row['avg_popularity'], row['norm'], str(int(row['k'])), fontsize=9)
            
        plt.title(f"Interest Key Norm vs. Avg Item Popularity (r={corr:.3f})")
        plt.xlabel("Average Item Popularity (Log Scale)")
        plt.ylabel("Interest Key L2 Norm")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plot_filename = "global_norm_vs_pop.png"
        plt.savefig(os.path.join(output_path, plot_filename))
        plt.close()
        
        report.add_figure(plot_filename, "Correlation Plot: Key Norm vs Popularity")
        
        # 데이터 CSV 저장 (나중에 논문용 그래프 다시 그릴 때 유용)
        df_stats.to_csv(os.path.join(output_path, "interest_stats_summary.csv"), index=False)

    # 7. 리포트 저장
    report.save(filename="deep_interest_analysis_report.md")
    print(f"Analysis complete. Report saved to {output_path}")

if __name__ == '__main__':
    EXPERIMENTS = [{'run_folder_path': 'trained_model/ml-1m/csar-bpr__negative_sampling_strategy=popularity'}]
    for exp in EXPERIMENTS:
        if os.path.exists(exp['run_folder_path']): run_full_analysis(exp)
        else: print(f"[Error] Path not found: {exp['run_folder_path']}")