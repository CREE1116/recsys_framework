import torch
import numpy as np
from src.data_loader import DataLoader
from src.models.csar.ASPIRE_Zero import ASPIRE_Zero
from src.evaluation import get_ndcg, get_recall

def verify_dataset(dataset_name, lambda_val):
    print(f"\nVerifying ASPIRE-Zero on {dataset_name} (lambda={lambda_val})...")
    
    config = {
        'ml1m': {
            'dataset_name': "ml-1m", 'data_path': "./data/ml1m/ratings.dat", 'separator': "::", 'columns': ["user_id", "item_id", "rating", "timestamp"],
            'rating_threshold': 0, 'min_user_interactions': 5, 'min_item_interactions': 5, 'split_method': "temporal_ratio", 'train_ratio': 0.8, 'valid_ratio': 0.1, 'data_cache_path': "./data_cache/"
        },
        'yahoo_r3': {
            'dataset_name': "yahoo_r3", 'train_file': "./data/yahooR3/processed/train_implicit_th0.txt", 'test_file': "./data/yahooR3/processed/test_implicit_th0.txt",
            'separator': "\t", 'columns': ["user_id", "item_id", "rating"], 'rating_threshold': 0, 'split_method': "presplit", 'data_cache_path': "./data_cache/"
        }
    }[dataset_name]

    dl = DataLoader(config)
    
    # Initialize Model with the new engine
    model_config = {
        'device': 'cpu',
        'dataset_name': dataset_name,
        'model': {
            'name': 'ASPIRE_Zero',
            'lambda_base': lambda_val,
            'max_iter': 50,
            'tol': 1e-5
        }
    }
    
    model = ASPIRE_Zero(model_config, dl)
    model.to('cpu') 
    
    # 1. Prepare User History for Masking
    user_history = dl.train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    # 2. Evaluation
    test_users = dl.test_df['user_id'].unique()
    scores = model.predict_full(torch.from_numpy(test_users))
    
    # 3. Mask Training Items (Crucial for correct metrics)
    for idx, u_id in enumerate(test_users):
        history = user_history.get(u_id, set())
        if history:
            seen_items = list(history)
            scores[idx, seen_items] = -1e9
            
    # Ground Truth
    ground_truth = dl.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # Get Top-20 Indices
    _, top_indices = torch.topk(scores, k=20, dim=1)
    top_indices = top_indices.cpu().numpy()
    
    ndcg_list = []
    recall_list = []
    
    for idx, u_id in enumerate(test_users):
        pred_list_20 = top_indices[idx].tolist()
        gt_list = ground_truth.get(u_id, [])
        
        ndcg_list.append(get_ndcg(pred_list_20, gt_list))
        recall_list.append(get_recall(pred_list_20, gt_list))
    
    ndcg_20 = np.mean(ndcg_list)
    recall_20 = np.mean(recall_list)
    
    print(f"[{dataset_name}] RESULTS:")
    print(f"  NDCG@20:   {ndcg_20:.4f}")
    print(f"  Recall@20: {recall_20:.4f}")
    return ndcg_20

if __name__ == "__main__":
    # Test on ML-1M (using a standard lambda=50.0)
    verify_dataset("ml1m", 50.0)
    
    # Test on Yahoo_R3 (using a standard lambda=10.0)
    verify_dataset("yahoo_r3", 10.0)
