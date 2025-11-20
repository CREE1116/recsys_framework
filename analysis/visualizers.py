
import torch
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



def generate_interest_report_section(report, interest_k, item_interests, inv_item_map, metadata_df, global_item_counts, top_n=15, interest_key_norm=None):
    """
    단일 관심사에 대한 심층 분석 리포트를 생성합니다.
    (그래프 생성 부분 제거됨, 빈 결과 에러 수정됨)
    """
    report.add_section(f"Interest #{interest_k} Analysis", level=3)
    
    stats_summary = {'k': interest_k}
    
    # 1. Interest Key L2 Norm
    if interest_key_norm is not None:
        if isinstance(interest_key_norm, torch.Tensor):
            norm_val = interest_key_norm.item()
        else:
            norm_val = interest_key_norm
        report.add_text(f"- **Interest Key L2 Norm:** `{norm_val:.4f}`")
        stats_summary['norm'] = norm_val
    
    # 2. Top-N Items Extraction
    weights_for_k = item_interests[:, interest_k]
    top_weights, top_indices = torch.topk(weights_for_k, top_n)
    
    top_items_info = []
    top_item_popularities = [] 
    
    for i, model_item_id in enumerate(top_indices.cpu().numpy()):
        original_item_id = inv_item_map.get(model_item_id)
        pop_count = global_item_counts.get(model_item_id, 0)
        top_item_popularities.append(pop_count)
        
        if original_item_id and original_item_id in metadata_df.index:
            item_info = metadata_df.loc[original_item_id]
            title_cleaned = re.sub(r'\s*\(\d{4}\)', '', item_info['title']).strip()
            
            top_items_info.append({
                'Rank': i + 1, 
                'Title': title_cleaned,
                'Year': int(item_info['year']) if pd.notna(item_info['year']) else None,
                'Genres': item_info['genres'], 
                'Popularity': pop_count,
                'Weight': top_weights[i].item()
            })
    
    # 3. Avg Popularity
    if top_item_popularities:
        avg_log_pop = np.mean(np.log1p(top_item_popularities))
        report.add_text(f"- **Avg Item Popularity (Log1p):** `{avg_log_pop:.4f}`")
        stats_summary['avg_popularity'] = avg_log_pop
    else:
        stats_summary['avg_popularity'] = 0.0

    if not top_items_info:
        report.add_text("Could not find metadata for top items.")
        return stats_summary

    top_items_df = pd.DataFrame(top_items_info)
    report.add_text(f"**Top {top_n} Items**")
    display_df = top_items_df[['Rank', 'Title', 'Year', 'Genres', 'Popularity', 'Weight']].copy()
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.4f}")
    report.add_table(display_df)

    # ---------------------------------------------------------
    # [수정됨] 통계적 유의성 검정 함수 (에러 방지 로직 추가)
    # ---------------------------------------------------------
    def perform_enrichment_test(target_counts, global_counts, top_n_size, total_size, feature_name):
        results = []
        for feature, count_in_top in target_counts.items():
            count_not_in_top = top_n_size - count_in_top
            global_count = global_counts.get(feature, 0)
            count_in_others = global_count - count_in_top
            total_others = total_size - top_n_size
            count_not_in_others = total_others - count_in_others
            
            contingency_table = [[count_in_top, count_not_in_top], [count_in_others, count_not_in_others]]
            chi2, p_value, _, _ = chi2_contingency(contingency_table, correction=True)
            
            top_ratio = count_in_top / top_n_size
            global_ratio = global_count / total_size if total_size > 0 else 0
            
            if p_value < 0.05 and top_ratio > global_ratio:
                lift = top_ratio / global_ratio if global_ratio > 0 else 0
                results.append({
                    feature_name: feature,
                    'Count': count_in_top,
                    'Top-N Ratio': f"{top_ratio:.1%}",
                    'Global Ratio': f"{global_ratio:.1%}",
                    'Lift': f"{lift:.1f}x",
                    'p-value': p_value # 정렬을 위해 float 상태 유지
                })
        
        # [FIX] 결과가 없을 경우 빈 DataFrame 반환 (여기서 에러가 났었음)
        if not results:
            return pd.DataFrame()

        # 결과가 있으면 DataFrame 생성 및 정렬
        df_res = pd.DataFrame(results)
        df_sorted = df_res.sort_values('p-value', ascending=True)
        
        # 출력용 포맷팅 (문자열 변환)
        df_sorted['p-value'] = df_sorted['p-value'].apply(lambda x: f"{x:.4f}")
        
        return df_sorted

    total_items_count = len(metadata_df)

    # 4. Genre Analysis
    report.add_section("Genre Analysis", level=4)
    current_genres = []
    for genres in top_items_df['Genres']:
        if isinstance(genres, str): current_genres.extend(genres.split('|'))
        elif isinstance(genres, list): current_genres.extend(genres)
    genre_counts = Counter(current_genres)
    
    all_genres_flat = []
    for genres in metadata_df['genres']:
        if isinstance(genres, str): all_genres_flat.extend(genres.split('|'))
        elif isinstance(genres, list): all_genres_flat.extend(genres)
    global_genre_counts = Counter(all_genres_flat)

    genre_enrichment_df = perform_enrichment_test(
        genre_counts, global_genre_counts, len(top_items_df), total_items_count, "Genre"
    )
    
    if not genre_enrichment_df.empty:
        report.add_text("**Statistically Significant Genres (p < 0.05)**")
        report.add_table(genre_enrichment_df)
    else:
        report.add_text("No specific genre is statistically over-represented.")

    # 5. Era (Decade) Analysis
    report.add_section("Era (Decade) Analysis", level=4)
    valid_years = [y for y in top_items_df['Year'] if pd.notna(y)]
    
    if valid_years:
        current_decades = [int(np.floor(y / 10) * 10) for y in valid_years]
        decade_counts = Counter(current_decades)
        
        all_years = pd.to_numeric(metadata_df['year'], errors='coerce').dropna()
        all_decades = [int(np.floor(y / 10) * 10) for y in all_years]
        global_decade_counts = Counter(all_decades)
        
        decade_enrichment_df = perform_enrichment_test(
            decade_counts, global_decade_counts, len(valid_years), len(all_years), "Decade"
        )
        
        report.add_text(f"- **Average Year:** {np.mean(valid_years):.1f}")
        
        if not decade_enrichment_df.empty:
            report.add_text("\n**Statistically Significant Decades (p < 0.05)**")
            report.add_table(decade_enrichment_df)
        else:
            report.add_text("No specific decade is statistically over-represented.")
    else:
        report.add_text("No year information available.")

    # 6. Qualitative Summary (그래프 생성 코드 제거됨)
    report.add_section("Qualitative Summary", level=4)
    summary_parts = []
    if not genre_enrichment_df.empty:
        summary_parts.append(f"**{genre_enrichment_df.iloc[0]['Genre']}** movies")
    if 'decade_enrichment_df' in locals() and not decade_enrichment_df.empty:
        summary_parts.append(f"from the **{decade_enrichment_df.iloc[0]['Decade']}s**")
        
    if summary_parts:
        summary = f"Interest #{interest_k} significantly focuses on " + " ".join(summary_parts) + "."
    else:
        summary = f"Interest #{interest_k} represents a diverse mix."
    report.add_text(summary)
    
    return stats_summary