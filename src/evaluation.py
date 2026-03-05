import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import json
from scipy.stats import gmean
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def get_hit_rate(pred_list, ground_truth):
    """
    Hit Rate (HR): 최소 하나라도 맞췄으면 1, 아니면 0.
    """
    intersection = set(pred_list).intersection(set(ground_truth))
    return 1 if len(intersection) > 0 else 0

def get_recall(pred_list, ground_truth):
    """
    Recall: (맞춘 개수 / 전체 정답 개수)
    """
    if len(ground_truth) == 0:
        return 0.0
    intersection = set(pred_list).intersection(set(ground_truth))
    return len(intersection) / len(ground_truth)

def get_precision(pred_list, ground_truth):
    """
    Precision: (맞춘 개수 / K)
    """
    if len(pred_list) == 0:
        return 0.0
    intersection = set(pred_list).intersection(set(ground_truth))
    return len(intersection) / len(pred_list)

def get_ndcg(pred_list, ground_truth):
    """
    NDCG for Multi-item setting.
    DCG = sum(1 / log2(rank + 2)) for each hit.
    IDCG = sum(1 / log2(i + 2)) for i in range(min(len(GT), K)).
    """
    if not ground_truth:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(pred_list):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
            
    # Calculate IDCG
    idcg = 0.0
    n_relevant = min(len(ground_truth), len(pred_list))
    for i in range(n_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def get_pop_ratio(target_item, item_popularity, mean_pop):
    """
    PopRatio: 맞춘 아이템의 정규화 인기도.
    """
    if hasattr(item_popularity, 'get'): # dict or Series
        target_pop = item_popularity.get(target_item, 1)
    else: # numpy array
        try:
            target_pop = item_popularity[target_item]
        except (IndexError, KeyError):
            target_pop = 1
            
    norm_pop = (target_pop / mean_pop) if mean_pop > 0 else 1.0
    return norm_pop

def get_coverage(all_recommended_items, n_items):
    """추천된 아이템의 고유 개수를 전체 아이템 수로 나눈 값."""
    if n_items == 0:
        return 0.0
    return len(set(all_recommended_items)) / n_items

def get_gini_index(values):
    """일반적인 Gini Index 계산 함수."""
    if len(values) == 0:
        return 0.0
    
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    
    if np.sum(values) == 0:
        return 0.0
        
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    return gini

def get_gini_index_from_recs(recommended_items, num_items):
    """
    추천된 아이템 목록의 Gini Index (전체 아이템 기준).
    추천되지 않은 아이템(count=0)도 포함하여 계산해야 정확한 불평등도를 측정할 수 있음.
    """
    if not recommended_items:
        return 0.0
    
    item_counts = np.zeros(num_items)
    for item in recommended_items:
        # 범위를 벗어나는 아이템 ID 방지
        if item < num_items:
            item_counts[item] += 1
    
    return get_gini_index(item_counts)

def get_novelty(recommended_items, item_popularity, num_users=None):
    """
    Novelty (Self-Information): -log2(P(i))
    P(i) = (train_count(i) + 1) / (total_interactions + N) (Smoothing 적용)
    
    Optimized: Uses numpy vectorization.
    """
    if not recommended_items:
        return 0.0
        
    recommended_items = np.array(recommended_items)
    
    # Convert item_popularity to lookup-friendly numpy array
    if isinstance(item_popularity, pd.Series):
        pop_values = item_popularity.values
    elif isinstance(item_popularity, dict):
        # Dict: create array up to max item id
        all_ids = list(item_popularity.keys())
        if recommended_items.size > 0:
            all_ids.append(recommended_items.max())
        max_item_id = max(all_ids)
        pop_values = np.zeros(max_item_id + 1)
        for k, v in item_popularity.items():
            pop_values[k] = v
    elif isinstance(item_popularity, np.ndarray):
        pop_values = item_popularity
    else:
        pop_values = np.array(item_popularity)

    total_interactions = pop_values.sum() if num_users is None else num_users
    n_items = len(pop_values)
    
    # Batch lookup
    counts = pop_values[recommended_items]
    p_i = (counts + 1) / (total_interactions + n_items)
    
    # -log2(p_i)
    novelty_scores = -np.log2(p_i + 1e-10)
    
    return np.mean(novelty_scores)

def get_long_tail_item_set(item_popularity, head_volume_percent=0.8):
    """
    롱테일 아이템 집합 반환 (Interaction Volume 기준).
    """
    if isinstance(item_popularity, np.ndarray):
        # Numpy array인 경우 Series로 전환하여 처리 (인덱스 유지를 위해)
        item_popularity = pd.Series(item_popularity)
        
    # 인기도 내림차순 정렬
    sorted_popularity = item_popularity.sort_values(ascending=False)
    
    # 누적합 계산
    cumsum = sorted_popularity.cumsum()
    total_interactions = sorted_popularity.sum()
    
    cutoff_val = total_interactions * head_volume_percent
    # cumsum이 cutoff_val보다 커지는 첫 번째 위치 찾기
    head_indices = np.searchsorted(cumsum.values, cutoff_val, side='right')
    
    # head_indices 개수만큼이 Head
    tail_item_ids = set(sorted_popularity.index[head_indices:].tolist())
    
    return tail_item_ids

def get_long_tail_coverage(all_recommended_items, item_popularity, head_volume_percent=0.8, precomputed_tail_set=None):
    if not all_recommended_items:
        return 0.0

    if precomputed_tail_set is not None:
        long_tail_item_ids = precomputed_tail_set
    else:
        # 일관성을 위해 get_long_tail_item_set 함수 사용
        long_tail_item_ids = get_long_tail_item_set(item_popularity, head_volume_percent)
    
    if len(long_tail_item_ids) == 0:
        return 0.0
    
    # Set operation is faster
    recs_set = set(all_recommended_items)
    intersection = recs_set.intersection(long_tail_item_ids)
    
    # Coverage = (추천된 롱테일 아이템 수) / (전체 롱테일 아이템 수)
    return len(intersection) / len(long_tail_item_ids)

def get_entropy_from_recs(recommended_items):
    if not recommended_items:
        return 0.0
    
    item_counts = {}
    for item in recommended_items:
        item_counts[item] = item_counts.get(item, 0) + 1
    
    popularity = np.array(list(item_counts.values()))
    if len(popularity) == 0:
        return 0.0
    
    probabilities = popularity / np.sum(popularity)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def get_gini_index_emb(item_embeddings=None, norms=None):
    """
    아이템 임베딩 Norm의 Gini Index.
    임베딩 공간 상에서 아이템들이 얼마나 고르게 분포(Magnitude 기준)되어 있는지 측정.
    일부 아이템(Popular)의 Norm만 비대해지는 현상을 감지할 수 있음.
    
    norms가 직접 제공되면 임베딩 계산을 생략함 (메모리 최적화).
    """
    if norms is not None:
        return get_gini_index(norms)
        
    if item_embeddings is None:
        return 0.0
    
    norms = torch.norm(item_embeddings, dim=1).detach().cpu().numpy()
    return get_gini_index(norms)

def get_ild(all_top_k_items, item_embeddings):
    """Intra-List Diversity = mean pairwise cosine distance over recommendations."""
    if not all_top_k_items:
        return 0.0

    all_ild_scores = []

    for user_recs in all_top_k_items:
        if len(user_recs) < 2:
            continue

        rec_item_embeds = item_embeddings[user_recs]
        if rec_item_embeds.dim() == 1:
            rec_item_embeds = rec_item_embeds.view(1, -1)

        # zero-norm 아이템이 있으면 F.normalize가 NaN을 만듦 → eps로 방어
        norms = rec_item_embeds.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        rec_item_embeds_norm = rec_item_embeds / norms

        cosine_sim_matrix = torch.matmul(rec_item_embeds_norm,
                                         rec_item_embeds_norm.transpose(0, 1))
        # nan 방어 (혹여 수치 문제가 있을 경우)
        cosine_sim_matrix = torch.nan_to_num(cosine_sim_matrix, nan=1.0)

        n = cosine_sim_matrix.size(0)
        # 상삼각 인덱스만 추출해서 pairwise distance 계산 (대각선 제외)
        row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=cosine_sim_matrix.device)
        pairwise_sims = cosine_sim_matrix[row_idx, col_idx]
        pairwise_divs = 1.0 - pairwise_sims

        num_pairs = pairwise_divs.numel()
        ild_score = pairwise_divs.mean().item() if num_pairs > 0 else 0.0
        all_ild_scores.append(ild_score)

    return float(np.mean(all_ild_scores)) if all_ild_scores else 0.0


def _evaluate_full(model, test_loader, top_k_list, metrics_list, device, user_history, item_popularity=None, long_tail_percent=0.8):
    """
    Refactored _evaluate_full for multi-item evaluation (Thesis-ready).
    1. Collects all data first to avoid dual iterator problem.
    2. Groups interactions by user for correct user-wise metrics.
    3. Implements Item-wise classification for Head/Tail metrics.
    """
    # 1. Collect all interaction pairs (vectorized for speed)
    print("[Evaluation] Collecting interaction pairs...")
    all_users_np = []
    all_items_np = []
    for user_batch, target_item_batch in test_loader:
        all_users_np.append(user_batch.numpy())
        all_items_np.append(target_item_batch.numpy())
    all_users_np = np.concatenate(all_users_np)
    all_items_np = np.concatenate(all_items_np)

    # 2. Aggregate ground truth items by user (vectorized via pandas groupby)
    import pandas as _pd
    _pairs_df = _pd.DataFrame({'u': all_users_np, 'i': all_items_np})
    user_test_ground_truth = _pairs_df.groupby('u')['i'].apply(list).to_dict()
            
    unique_test_users = sorted(list(user_test_ground_truth.keys()))
    
    # Metrics results storage
    results = {f'{metric}@{k}': [] for k in top_k_list for metric in metrics_list}
    all_top_k_items = [] 
    
    pop_ratio_raw = {k: [] for k in top_k_list}
    mean_pop = None
    if 'PopRatio' in metrics_list and item_popularity is not None:
        if hasattr(item_popularity, 'get'):
            all_pops = [item_popularity.get(i, 1) for i in range(len(item_popularity))]
        else: # numpy array
            all_pops = item_popularity
        mean_pop = np.mean(all_pops)
    
    tail_item_set = None
    need_tail_metrics = any(m in metrics_list for m in ['LongTailHitRate', 'LongTailNDCG', 'HeadHitRate', 'HeadNDCG'])
    if need_tail_metrics and item_popularity is not None:
        tail_item_set = get_long_tail_item_set(item_popularity, head_volume_percent=long_tail_percent)

    if not top_k_list:
        return {}, []

    # 3. Iterate by Batch of Unique Users
    batch_size = test_loader.batch_size
    mask_after_topk = getattr(model, 'mask_after_topk', False)
    k_target_max = max(top_k_list)
    k_fetch = k_target_max + 500 if mask_after_topk else k_target_max
    k_fetch = min(k_fetch, model.n_items)

    import sys
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_test_users), batch_size), desc="Evaluating (user-wise)", leave=False, dynamic_ncols=True, file=sys.stdout):
            user_batch_ids = unique_test_users[i:i+batch_size]
            user_tensor = torch.LongTensor(user_batch_ids).to(device)
            
            all_item_scores = model.forward(user_tensor) # [B, N_ITEMS]
            
            # [Standard Masking]
            if not mask_after_topk:
                for idx, u_id in enumerate(user_batch_ids):
                    items_seen = user_history.get(u_id, set())
                    if items_seen:
                        target_items_set = set(user_test_ground_truth[u_id])
                        to_exclude = list(items_seen - target_items_set)
                        if to_exclude:
                            all_item_scores[idx, to_exclude] = -1e9

            _, top_indices = torch.topk(all_item_scores, k=k_fetch, dim=1)
            del all_item_scores
            
            top_indices_cpu = top_indices.cpu().numpy()
            
            for idx, u_id in enumerate(user_batch_ids):
                pred_list = top_indices_cpu[idx].tolist()
                ground_truth = user_test_ground_truth[u_id]
                
                if mask_after_topk:
                    items_seen = user_history.get(u_id, set())
                    to_exclude = {it for it in items_seen if it not in ground_truth}
                    pred_list = [it for it in pred_list if it not in to_exclude][:k_target_max]
                else:
                    pred_list = pred_list[:k_target_max]
                
                all_top_k_items.append(pred_list)
                
                # Metric calculation per user
                for k in top_k_list:
                    pred_list_k = pred_list[:k]
                    
                    if 'HitRate' in metrics_list:
                        results[f'HitRate@{k}'].append(get_hit_rate(pred_list_k, ground_truth))
                    if 'Recall' in metrics_list:
                        results[f'Recall@{k}'].append(get_recall(pred_list_k, ground_truth))
                    if 'Precision' in metrics_list:
                        results[f'Precision@{k}'].append(get_precision(pred_list_k, ground_truth))
                    if 'NDCG' in metrics_list:
                        results[f'NDCG@{k}'].append(get_ndcg(pred_list_k, ground_truth))
                    
                    # 4. Item-wise Refinement for LongTail/Head (Thesis Style)
                    if need_tail_metrics and tail_item_set is not None:
                        # Tail Items Metrics
                        tail_gt = [it for it in ground_truth if it in tail_item_set]
                        if tail_gt:
                            if 'LongTailHitRate' in metrics_list:
                                results[f'LongTailHitRate@{k}'].append(get_hit_rate(pred_list_k, tail_gt))
                            if 'LongTailNDCG' in metrics_list:
                                results[f'LongTailNDCG@{k}'].append(get_ndcg(pred_list_k, tail_gt))
                        
                        # Head Items Metrics
                        head_gt = [it for it in ground_truth if it not in tail_item_set]
                        if head_gt:
                            if 'HeadHitRate' in metrics_list:
                                results[f'HeadHitRate@{k}'].append(get_hit_rate(pred_list_k, head_gt))
                            if 'HeadNDCG' in metrics_list:
                                results[f'HeadNDCG@{k}'].append(get_ndcg(pred_list_k, head_gt))
                    
                    # PopRatio
                    if 'PopRatio' in metrics_list and mean_pop is not None:
                        hits = [it for it in pred_list_k if it in ground_truth]
                        if hits:
                            pops = [get_pop_ratio(it, item_popularity, mean_pop) for it in hits]
                            pop_ratio_raw[k].append(np.mean(pops))

    # Calculate final averages (using nanmean to ignore None/empty entries)
    final_results = {key: np.nanmean(value) if value else 0.0 for key, value in results.items()}
    
    if 'PopRatio' in metrics_list:
        for k in top_k_list:
            final_results[f'PopRatio@{k}'] = np.mean(pop_ratio_raw[k]) if pop_ratio_raw[k] else 0.0
    
    return final_results, all_top_k_items

def _evaluate_uni99(model, test_loader, top_k_list, metrics_list, device):
    results = {f'{metric}@{k}': [] for k in top_k_list for metric in metrics_list}
    all_top_k_items = []
    use_fast_path = all(m in ['HitRate', 'NDCG'] for m in metrics_list)

    # top_k_list가 비어있으면 평가할 필요 없음
    if not top_k_list:
        return {}, []
        
    import sys
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (uni99)", leave=False, dynamic_ncols=True, file=sys.stdout):
            
            user_ids = batch['user_id'].to(device)
            item_ids = batch['items'].to(device)
            batch_size = user_ids.size(0)

            user_ids_flat = user_ids.repeat_interleave(100)
            item_ids_flat = item_ids.view(-1)
            
            all_scores = model.predict_for_pairs(user_ids_flat, item_ids_flat)
            scores_by_user = all_scores.view(batch_size, 100)

            if use_fast_path:
                target_scores = scores_by_user[:, 0].unsqueeze(1)
                num_better_negatives = (scores_by_user[:, 1:] >= target_scores).sum(dim=1)
                ranks = 1 + num_better_negatives

                for k in top_k_list:
                    in_top_k = (ranks <= k)
                    if 'HitRate' in metrics_list:
                        results[f'HitRate@{k}'].extend(in_top_k.cpu().numpy())
                    if 'NDCG' in metrics_list:
                        ndcg_scores = torch.zeros_like(ranks, dtype=torch.float)
                        ndcg_scores[in_top_k] = 1.0 / torch.log2(ranks[in_top_k].float() + 1)
                        results[f'NDCG@{k}'].extend(ndcg_scores.cpu().numpy())
            else:
                _, top_indices_100 = torch.topk(scores_by_user, k=100, dim=1)
                
                # torch>=2.9: MPS gather 네이티브 지원
                pred_lists_100 = torch.gather(item_ids, 1, top_indices_100).cpu().numpy().tolist()
                target_items = item_ids[:, 0].cpu().numpy().tolist()

                all_top_k_items.extend([p[:max(top_k_list)] for p in pred_lists_100])

                for i in range(batch_size):
                    pred_list_100 = pred_lists_100[i]
                    target_item = target_items[i]
                    for k in top_k_list:
                        pred_list_k = pred_list_100[:k]
                        if 'HitRate' in metrics_list:
                            results[f'HitRate@{k}'].append(get_hit_rate(pred_list_k, target_item))
                        if 'NDCG' in metrics_list:
                            results[f'NDCG@{k}'].append(get_ndcg(pred_list_k, target_item))

    return {key: np.mean(value) if value else 0.0 for key, value in results.items()}, all_top_k_items


def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    """
    [BUG FIX] The `config` parameter is now treated as the evaluation config directly.
    Defaults for `metrics` and `top_k` are now empty lists to enforce strict YAML configuration.
    """
    method = eval_config.get('method', 'full')
    top_k_list = eval_config.get('top_k', [])
    metrics_list = eval_config.get('metrics', [])
    
    model.eval()
    
    if method == 'full' or method == 'full_subset' or method == 'sampled':
        # [RecBole Alignment] Validation 시에는 train 마스킹, Test 시에는 train+valid 마스킹
        history_to_use = data_loader.eval_user_history if is_final else data_loader.train_user_history
        # PopRatio, LongTail 지표 계산을 위해 item_popularity 전달
        need_pop = any(m in metrics_list for m in ['PopRatio', 'LongTailHitRate', 'LongTailNDCG', 'HeadHitRate', 'HeadNDCG', 'Recall'])
        item_pop = data_loader.item_popularity if need_pop else None
        
        lt_percent = eval_config.get('long_tail_percent', 0.8)
        core_metrics, all_top_k_items = _evaluate_full(model, test_loader, top_k_list, metrics_list, device, history_to_use, item_pop, lt_percent)
    elif method == 'uni99':
        core_metrics, all_top_k_items = _evaluate_uni99(model, test_loader, top_k_list, metrics_list, device)
    else:
        # metrics_list가 비어있으면 평가를 건너뛰고 빈 결과를 반환
        if not metrics_list:
            return {}
        raise ValueError(f"Unknown evaluation method: {method}")

    final_results = core_metrics
    
    if not any(m in metrics_list for m in ['Coverage', 'GiniIndex', 'LongTailCoverage', 'Entropy', 'ILD', 'GiniIndex_emb', 'Novelty']):
        return final_results

    item_embeddings_for_ild = None
    all_item_norms = None
    if any(m in metrics_list for m in ['ILD', 'GiniIndex_emb']):
        with torch.no_grad():
            if hasattr(model, 'get_embeddings'):
                _, item_embeddings_for_ild = model.get_embeddings()
            
            # Fallback for models without traditional embeddings (e.g. EASE, ItemKNN)
            if item_embeddings_for_ild is None and hasattr(model, 'get_final_item_embeddings'):
                item_embeddings_for_ild = model.get_final_item_embeddings()
            
            # Last fallback: item_embedding.weight
            if item_embeddings_for_ild is None and hasattr(model, 'item_embedding') and hasattr(model.item_embedding, 'weight'):
                item_embeddings_for_ild = model.item_embedding.weight
        
        # [메모리 최적화] GiniIndex_emb@k 계산을 위해 미리 Norm을 뽑아둠
        if 'GiniIndex_emb' in metrics_list and item_embeddings_for_ild is not None:
            all_item_norms = torch.norm(item_embeddings_for_ild, dim=1).detach()

    # [Global Metrics] GiniIndex, GiniIndex_emb
    # 전체 Top-K(최대 K) 기준 글로벌 지표도 남겨둠
    if 'GiniIndex' in metrics_list:
        flat_recommended_items_overall = [item for sublist in all_top_k_items for item in sublist]
        final_results['GiniIndex'] = get_gini_index_from_recs(flat_recommended_items_overall, data_loader.n_items)
        
    if 'GiniIndex_emb' in metrics_list and all_item_norms is not None:
        # 전체 아이템 임베딩의 Gini Index (Static) - "모델 자체의 표현력 불균형"
        final_results['GiniIndex_emb'] = get_gini_index_emb(norms=all_item_norms.cpu().numpy())

    for k in top_k_list:
        all_top_k_items_at_k = [user_recs[:k] for user_recs in all_top_k_items]
        flat_recommended_items_at_k = [item for sublist in all_top_k_items_at_k if sublist for item in sublist]
        
        if 'ILD' in metrics_list and item_embeddings_for_ild is not None:
            final_results[f'ILD@{k}'] = get_ild(all_top_k_items_at_k, item_embeddings_for_ild)
        
        if 'Coverage' in metrics_list:
            final_results[f'Coverage@{k}'] = get_coverage(flat_recommended_items_at_k, data_loader.n_items)
            
        if 'LongTailCoverage' in metrics_list:
            long_tail_percent = eval_config.get('long_tail_percent', 0.8)
            # Pre-compute if not already done in _evaluate_full (but _evaluate_full does not return it)
            # Efficiently compute once here
            tail_set = get_long_tail_item_set(data_loader.item_popularity, head_volume_percent=long_tail_percent)
            final_results[f'LongTailCoverage@{k}'] = get_long_tail_coverage(flat_recommended_items_at_k, data_loader.item_popularity, long_tail_percent, precomputed_tail_set=tail_set)
            
        if 'Entropy' in metrics_list:
            final_results[f'Entropy@{k}'] = get_entropy_from_recs(flat_recommended_items_at_k)
            
        if 'Novelty' in metrics_list:
             final_results[f'Novelty@{k}'] = get_novelty(flat_recommended_items_at_k, data_loader.item_popularity)
             
        if 'GiniIndex' in metrics_list:
             final_results[f'GiniIndex@{k}'] = get_gini_index_from_recs(flat_recommended_items_at_k, data_loader.n_items)

        if 'GiniIndex_emb' in metrics_list and all_item_norms is not None:
             # 추천된 아이템들의 임베딩 Norm Gini Index
             # [메모리 최적화] 전체 임베딩을 인덱싱하지 않고 미리 계산된 Norm만 인덱싱함
             if flat_recommended_items_at_k:
                 flat_indices = torch.tensor(flat_recommended_items_at_k, device=all_item_norms.device)
                 recs_norms = all_item_norms[flat_indices].cpu().numpy()
                 final_results[f'GiniIndex_emb@{k}'] = get_gini_index_emb(norms=recs_norms)
             else:
                 final_results[f'GiniIndex_emb@{k}'] = 0.0

    return final_results
