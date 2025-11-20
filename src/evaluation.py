import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import json
from scipy.stats import gmean
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def get_hit_rate(pred_list, target_item):
    return 1 if target_item in pred_list else 0

def get_ndcg(pred_list, target_item):
    if target_item in pred_list:
        rank = pred_list.index(target_item)
        return np.log(2) / np.log(rank + 2)
    return 0

def get_coverage(all_recommended_items, n_items):
    """추천된 아이템의 고유 개수를 전체 아이템 수로 나눈 값."""
    return len(set(all_recommended_items)) / n_items

def get_gini_index_from_recs(recommended_items):
    """추천된 아이템 목록의 지니 계수."""
    if not recommended_items:
        return 0.0
    
    item_counts = {}
    for item in recommended_items:
        item_counts[item] = item_counts.get(item, 0) + 1
    
    popularity = np.array(list(item_counts.values()))
    if len(popularity) == 0:
        return 0.0
    
    popularity = np.sort(popularity)
    n = len(popularity)
    index = np.arange(1, n + 1)
    
    gini = (np.sum((2 * index - n - 1) * popularity)) / (n * np.sum(popularity))
    return gini

def get_long_tail_coverage(all_recommended_items, item_popularity, long_tail_percent=0.2):
    if not all_recommended_items:
        return 0.0

    sorted_popularity = item_popularity.sort_values(ascending=True)
    num_long_tail_items = int(len(sorted_popularity) * long_tail_percent)
    long_tail_item_ids = set(sorted_popularity.index[:num_long_tail_items].tolist())

    recommended_long_tail_items = [item for item in all_recommended_items if item in long_tail_item_ids]
    
    return len(set(recommended_long_tail_items)) / len(set(all_recommended_items)) if len(set(all_recommended_items)) > 0 else 0.0

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

def get_ild(all_top_k_items, item_embeddings):
    if not all_top_k_items:
        return 0.0

    all_ild_scores = []

    for user_recs in all_top_k_items:
        if len(user_recs) < 2:
            continue

        rec_item_embeds = item_embeddings[user_recs]
        rec_item_embeds_norm = F.normalize(rec_item_embeds, p=2, dim=1)

        # [최적화] 전체 쌍 유사도는 한 번에 계산
        cosine_sim_matrix = torch.matmul(rec_item_embeds_norm,
                                         rec_item_embeds_norm.transpose(0, 1))

        n = cosine_sim_matrix.size(0)
        if n < 2:
            continue

        # 대각선(자기 자신)은 0으로
        cosine_sim_matrix.fill_diagonal_(0.0)

        # 상삼각(or 하삼각)만 사용해서 중복 제거
        triu_indices = torch.triu_indices(n, n, offset=1)
        pair_sims = cosine_sim_matrix[triu_indices[0], triu_indices[1]]

        num_pairs = pair_sims.numel()
        if num_pairs == 0:
            continue

        # diversity = 1 - cos_sim
        diversity_vals = 1.0 - pair_sims
        all_ild_scores.append(diversity_vals.mean().item())

    return float(np.mean(all_ild_scores)) if all_ild_scores else 0.0


def _evaluate_full(model, test_loader, top_k_list, metrics_list, device, user_history):
    results = {f'{metric}@{k}': [] for k in top_k_list for metric in metrics_list}
    all_top_k_items = [] 

    # top_k_list가 비어있으면 평가할 필요 없음
    if not top_k_list:
        return {}, []

    with torch.no_grad():
        for user_batch, target_item_batch in tqdm(test_loader, desc="Evaluating (full, batched)"):
            user_batch = user_batch.to(device)
            all_item_scores = model.forward(user_batch)

            # 한 번만 CPU로 내리고 재사용
            user_batch_cpu = user_batch.cpu().numpy()

            for i, user_id in enumerate(user_batch_cpu):
                items = user_history.get(user_id)
                if items is None:
                    continue

                target_item_id = target_item_batch[i].item()

                # target을 빼고 마스크할 아이템만 만들기
                if target_item_id in items:
                    to_mask = [it for it in items if it != target_item_id]
                else:
                    to_mask = list(items)

                if to_mask:
                    all_item_scores[i, to_mask] = -torch.inf


            _, top_indices = torch.topk(all_item_scores, k=max(top_k_list), dim=1)
            
            for i in range(len(user_batch)):
                pred_list = top_indices[i].cpu().numpy().tolist()
                target_item = target_item_batch[i].item()
                all_top_k_items.append(pred_list)

                for k in top_k_list:
                    pred_list_k = pred_list[:k]
                    if 'HitRate' in metrics_list:
                        results[f'HitRate@{k}'].append(get_hit_rate(pred_list_k, target_item))
                    if 'NDCG' in metrics_list:
                        results[f'NDCG@{k}'].append(get_ndcg(pred_list_k, target_item))

    return {key: np.mean(value) if value else 0.0 for key, value in results.items()}, all_top_k_items

def _evaluate_uni99(model, test_loader, top_k_list, metrics_list, device):
    results = {f'{metric}@{k}': [] for k in top_k_list for metric in metrics_list}
    all_top_k_items = []
    use_fast_path = all(m in ['HitRate', 'NDCG'] for m in metrics_list)

    # top_k_list가 비어있으면 평가할 필요 없음
    if not top_k_list:
        return {}, []
        
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (uni99)"):
            
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
                
                pred_lists_100 = torch.gather(item_ids, 1, top_indices_100.cpu()).numpy().tolist()
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


def evaluate_metrics(model, data_loader, eval_config, device, test_loader):
    """
    [BUG FIX] The `config` parameter is now treated as the evaluation config directly.
    Defaults for `metrics` and `top_k` are now empty lists to enforce strict YAML configuration.
    """
    method = eval_config.get('method', 'full')
    top_k_list = eval_config.get('top_k', [])
    metrics_list = eval_config.get('metrics', [])
    
    model.eval()
    
    if method == 'full':
        core_metrics, all_top_k_items = _evaluate_full(model, test_loader, top_k_list, metrics_list, device, data_loader.user_history)
    elif method == 'uni99':
        core_metrics, all_top_k_items = _evaluate_uni99(model, test_loader, top_k_list, metrics_list, device)
    else:
        # metrics_list가 비어있으면 평가를 건너뛰고 빈 결과를 반환
        if not metrics_list:
            return {}
        raise ValueError(f"Unknown evaluation method: {method}")

    final_results = core_metrics
    
    if not any(m in metrics_list for m in ['Coverage', 'GiniIndex', 'LongTailCoverage', 'Entropy', 'ILD']):
        return final_results

    item_embeddings_for_ild = None
    if 'ILD' in metrics_list:
        with torch.no_grad():
            if hasattr(model, 'get_embeddings'):
                _, item_embeddings_for_ild = model.get_embeddings()
            elif hasattr(model, 'item_embedding') and hasattr(model.item_embedding, 'weight'):
                item_embeddings_for_ild = model.item_embedding.weight

    # [수정] GiniIndex는 K값과 무관하게 전체 추천 목록에 대해 한 번만 계산
    if 'GiniIndex' in metrics_list:
        flat_recommended_items_overall = [item for sublist in all_top_k_items for item in sublist]
        final_results['GiniIndex'] = get_gini_index_from_recs(flat_recommended_items_overall)

    for k in top_k_list:
        all_top_k_items_at_k = [user_recs[:k] for user_recs in all_top_k_items]
        
        if 'ILD' in metrics_list and item_embeddings_for_ild is not None:
            final_results[f'ILD@{k}'] = get_ild(all_top_k_items_at_k, item_embeddings_for_ild)
        
        # [BUG FIX] Coverage, Entropy 등은 K값에 따라 계산되어야 함
        flat_recommended_items_at_k = [item for sublist in all_top_k_items_at_k if sublist for item in sublist]
        if 'Coverage' in metrics_list:
            final_results[f'Coverage@{k}'] = get_coverage(flat_recommended_items_at_k, data_loader.n_items)
        if 'LongTailCoverage' in metrics_list:
            long_tail_percent = eval_config.get('long_tail_percent', 0.2)
            final_results[f'LongTailCoverage@{k}'] = get_long_tail_coverage(flat_recommended_items_at_k, data_loader.item_popularity, long_tail_percent)
        if 'Entropy' in metrics_list:
            final_results[f'Entropy@{k}'] = get_entropy_from_recs(flat_recommended_items_at_k)
    return final_results
