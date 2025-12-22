import os
import argparse
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from analysis.utils import load_model_from_run, get_analysis_output_path, AnalysisReport, load_item_metadata


# --------------------------------------------------------------------------------
# 1. 데이터 로드 함수 (증강된 메타데이터 로드용)
# --------------------------------------------------------------------------------
def load_augmented_metadata(data_dir, dataset_name='ml-1m'):
    """ 
    우선순위:
    1. movies_augmented.csv (Rich Metadata for ML-1M)
    2. analysis.utils.load_item_metadata (Generic/Amazon support)
    """
    
    # [핵심 수정] data_dir가 파일(ratings.dat)을 가리키면 부모 폴더 경로로 보정
    if os.path.isfile(data_dir):
        data_dir = os.path.dirname(data_dir)

    # 1. Try Augmented Metadata (Custom for ML-1M)
    aug_path = os.path.join(data_dir, 'movies_augmented.csv')
    if os.path.exists(aug_path):
        print(f"Loading augmented metadata from {aug_path}")
        df = pd.read_csv(aug_path)
        df = df.fillna('')
        return df.set_index('item_id')
    
    # 2. Fallback to Standard/Amazon Metadata
    print(f"[Info] Augmented metadata not found. Falling back to standard metadata for '{dataset_name}'.")
    
    # Re-construct file path logic if needed, but utils.load_item_metadata handles generic paths well
    # We pass the directory or file path as passed to this function (which originates from config['data_path'])
    # utils.load_item_metadata expects 'data_path' (usually the interaction file), 
    # but here we might have a dir. Let's rely on utils logic which handles dirname internally.
    # However, utils.load_item_metadata expects the FULL path to interaction file usually.
    # Let's pass 'data_dir' joined with a dummy file if needed, or rely on it.
    # Actually utils.load_item_metadata does `os.path.dirname(data_path)` internally.
    # So if we pass a directory, dirname(directory) might be parent.
    # To be safe, let's pass a dummy file path inside this dir so dirname works, 
    # OR better: call load_item_metadata with a constructed path that works.
    
    # Actually, simpler: load_item_metadata is robust.
    # But wait, config['data_path'] is usually '.../ratings.dat'. 
    # Here `data_dir` is already `dirname`.
    # Let's just pass `os.path.join(data_dir, 'dummy_file')` so utils can strip it.
    # Or just fix utils to handle directories? No, don't touch utils now if possible.
    
    dummy_path = os.path.join(data_dir, 'dummy_interaction_file.dat')
    return load_item_metadata(dataset_name, dummy_path)

# --------------------------------------------------------------------------------
# 2. 핵심 분석 함수 (장르/연도 + 감독/배우/키워드 추가)
# --------------------------------------------------------------------------------
def generate_deep_interest_report(report, interest_k, item_interests, inv_item_map, metadata_df, global_stats, top_n=20, interest_key_norm=None, view_name="Interest"):
    """
    모든 메타데이터(장르, 연도, 감독, 배우, 키워드)에 대해 과잉 표집 분석을 수행합니다.
    """
    report.add_section(f"{view_name} #{interest_k} Deep Analysis", level=3)
    
    stats_summary = {'k': interest_k, 'view': view_name}

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
        'genres': [], 'decades': [], 'directors': [], 'cast': [], 'keywords': [], 'brands': [] # [NEW] Brand (Author)
    }

    for i, model_item_id in enumerate(top_indices.cpu().numpy()):
        original_item_id = inv_item_map.get(model_item_id)
        pop_count = global_stats['item_counts'].get(model_item_id, 0)
        top_item_pops.append(pop_count)
        
        if original_item_id and original_item_id in metadata_df.index:
            row = metadata_df.loc[original_item_id]
            
            # 제목 정제
            title = str(row.get('title', ''))
            title_cleaned = re.sub(r'\s*\(\d{4}\)', '', title).strip()
            
            # 연도/연대 처리
            year = None
            match = re.search(r'\((\d{4})\)', title)
            if match:
                year = int(match.group(1))
                current_features['decades'].append(int(np.floor(year / 10) * 10))

            # 장르 처리
            raw_genres = row['genres']
            if isinstance(raw_genres, list):
                genres = raw_genres
            elif isinstance(raw_genres, str):
                genres = raw_genres.split('|') if raw_genres else []
            else:
                genres = []
            
            current_features['genres'].extend(genres)

            # [Director] or [Brand/Author] (for Books)
            director = row.get('director', '')
            brand = row.get('brand', '') # [NEW]
            
            if director: current_features['directors'].append(director)
            if brand: current_features['brands'].append(brand)

            # [추가] 배우 처리
            cast = row.get('cast', '').split('|') if row.get('cast') else []
            current_features['cast'].extend(cast)

            # [추가] 키워드 처리 (Movies: keywords column / Books: extract from description?)
            kws = row.get('keywords', '').split('|') if row.get('keywords') else []
            
            # Fallback for books: simple keyword extraction from description/title if keywords empty
            if not kws and (row.get('description') or title_cleaned):
                text = (str(row.get('description', '')) + ' ' + title_cleaned).lower()
                # Simple extraction: words > 4 chars, not common stop words (very basic)
                words = re.findall(r'\b[a-z]{5,}\b', text)
                # Filter out some very common words if needed (optional)
                kws = list(set(words))
                
            current_features['keywords'].extend(kws)

            top_items_info.append({
                'Rank': i + 1, 'Title': title_cleaned, 'Year': year, 'Genres': ', '.join(genres[:2]),
                'Director/Author': director if director else brand, 
                'Weight': f"{top_weights[i].item():.4f}"
            })

    # 테이블 출력
    if top_items_info:
        df_show = pd.DataFrame(top_items_info)
        report.add_text(f"**Top {top_n} Items**")
        
        # Dynamic column selection
        cols_to_show = ['Rank', 'Title', 'Year', 'Genres', 'Weight']
        if 'Director/Author' in df_show.columns and df_show['Director/Author'].any():
            cols_to_show.insert(4, 'Director/Author')
            
        report.add_table(df_show[cols_to_show])

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
    
    # [NEW] Brand (Author)
    top_brand = run_test(current_features['brands'], global_stats['brands'], "Authors/Brands", min_count=2)
    
    # (3) 배우 (Cast)
    top_cast = run_test(current_features['cast'], global_stats['cast'], "Cast", min_count=3)
    
    # (4) 키워드 (Keywords)
    top_keyword = run_test(current_features['keywords'], global_stats['keywords'], "Keywords", min_count=3)

    # ---------------------------------------------------------
    # 4. 정성적 요약 (자동 생성)
    # ---------------------------------------------------------
    report.add_section("Qualitative Summary", level=4)
    summary = f"{view_name} #{interest_k} shows specific preferences."
    
    points = []
    if top_director: points.append(f"films by **{top_director}**")
    if top_brand: points.append(f"books by **{top_brand}**") # [NEW]
    if top_genre: points.append(f"**{top_genre}** genre")
    if top_decade: points.append(f"from the **{top_decade}s**")
    if top_keyword: points.append(f"related to **'{top_keyword}'**")
    
    if points:
        summary = f"{view_name} #{interest_k} captures **{', '.join(points)}**."
    
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
    metadata_df = load_augmented_metadata(config['data_path'], config['dataset_name'])
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}
    
    # [Optimization] Filter metadata to only items in the dataset
    # metadata_df index is 'item_id' (original string ID)
    # inv_item_map.values() contains original user/item IDs? No, inv_item_map[model_id] = original_id
    valid_original_ids = set(inv_item_map.values())
    
    # Filter metadata (Intersection of metadata and dataset items)
    original_len = len(metadata_df)
    metadata_df = metadata_df[metadata_df.index.isin(valid_original_ids)]
    print(f"[Optimization] Filtered metadata from {original_len} to {len(metadata_df)} items (matching dataset).")
    
    # Deduplicate metadata index to prevent duplicate lookups
    if metadata_df.index.duplicated().any():
        print(f"[Info] Deduplicating metadata index. Found {metadata_df.index.duplicated().sum()} duplicates.")
        metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
    
    # 전역 통계 계산 (비교군 생성을 위해 미리 한 번 훑기)
    print("Calculating global statistics...")
    global_stats = {
        'item_counts': data_loader.df['item_id'].value_counts().to_dict(),
        'genres': Counter(), 'decades': Counter(), 
        'directors': Counter(), 'cast': Counter(), 'keywords': Counter(), 'brands': Counter() # [NEW]
    }
    
    for idx, row in metadata_df.iterrows():
        # Genre
        if row.get('genres'): 
            g_val = row['genres']
            if isinstance(g_val, list): global_stats['genres'].update(g_val)
            elif isinstance(g_val, str): global_stats['genres'].update(g_val.split('|'))
        # Decade
        # Decade
        match = re.search(r'\((\d{4})\)', str(row.get('title', '')))
        if match: global_stats['decades'].update([int(np.floor(int(match.group(1))/10)*10)])
        # Director
        if row.get('director'): global_stats['directors'].update([row['director']])
        # Brand (Author) [NEW]
        if row.get('brand'): global_stats['brands'].update([row['brand']])
        # Cast
        if row.get('cast'): global_stats['cast'].update(str(row['cast']).split('|'))
        # Keywords (Fallback to description logic handled in generating report, but for global stats we stick to explicit keywords if any)
        # Note: If we use description extraction globally, it would be too slow here. 
        # So we only use explicit keywords for global stats, or accept that 'significance' test for description-keywords is approximate (using explicit only as background).
        if row.get('keywords'): global_stats['keywords'].update(str(row['keywords']).split('|'))

    # 리포트 생성
    run_name = os.path.basename(run_folder_path)
    output_path = get_analysis_output_path(config['dataset_name'], run_name)
    report = AnalysisReport(f"Deep Semantic Analysis: {run_name}", output_path)
    
    # 분석 루프
    with torch.no_grad():
        attention_output = model.attention_layer(model.item_embedding.weight)
        
        # [DualView Check]
        if isinstance(attention_output, tuple):
            print("[Info] Dual-View Model Detected. Analyzing both Like and Dislike views.")
            # DualView: (like_interests, dislike_interests)
            item_interests_tuple = attention_output
            # Keys: model.attention_layer.pos_keys, neg_keys
            keys_tuple = (model.attention_layer.pos_keys, model.attention_layer.neg_keys)
            view_names = ["Like Interest", "Dislike Interest"]
        else:
            # Single View
            item_interests_tuple = (attention_output,)
            # Keys: model.attention_layer.interest_keys usually, but handle Dummy/Tensor cases generic
            # Check if interest_keys exists (Standard CSAR)
            if hasattr(model.attention_layer, 'interest_keys'):
                keys_tuple = (model.attention_layer.interest_keys,)
            else:
                # Fallback if no interest_keys param found (unlikely for standard)
                keys_tuple = (None,)
            view_names = ["Interest"]

    collected_stats = []
    
    for i, item_interests in enumerate(item_interests_tuple):
        view_name = view_names[i]
        curr_keys = keys_tuple[i]
        
        # Calculate Norms if keys exist
        keys_norm = None
        if curr_keys is not None:
            # Handle if keys is parameter/tensor and handle Dummy (extra key)
            if hasattr(model.attention_layer, 'Dummy') and model.attention_layer.Dummy and i==0: # Only for single view Dummy usually
                 curr_keys_real = curr_keys[:-1] 
            else:
                 curr_keys_real = curr_keys
            
            keys_norm = torch.norm(curr_keys_real, p=2, dim=1).cpu()

        report.add_section(f"=== {view_name} View Analysis ===", level=1)
        
        for k in range(model.num_interests):
            k_norm = keys_norm[k] if keys_norm is not None else None
            
            stats = generate_deep_interest_report(
                report, k, item_interests, inv_item_map, metadata_df, global_stats, top_n, k_norm, view_name
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
    parser = argparse.ArgumentParser(description="Run Deep Semantic Analysis on CSAR model interests.")
    parser.add_argument('--exp_dir', type=str, help='Path to the experiment directory.')
    
    args = parser.parse_args()

    if args.exp_dir:
        # Run specific experiment from CLI
        if os.path.exists(args.exp_dir):
            run_full_analysis({'run_folder_path': args.exp_dir})
        else:
            print(f"[Error] Path not found: {args.exp_dir}")
    else:
        # Default behavior (Hardcoded list - optional, kept for backward compat)
        EXPERIMENTS = [{'run_folder_path': '/Users/leejongmin/code/recsys_framework/trained_model/ml-1m/csar-hard'}]
        for exp in EXPERIMENTS:
            if os.path.exists(exp['run_folder_path']): run_full_analysis(exp)
            else: print(f"[Error] Path not found: {exp['run_folder_path']}")