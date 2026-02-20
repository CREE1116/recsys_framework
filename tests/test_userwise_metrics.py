import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import get_hit_rate, get_recall, get_precision, get_ndcg

def test_metrics():
    print("Testing User-wise Metrics...")
    
    # User 1: 3 ground truth items, 5 recommendations, 2 hits
    ground_truth = [1, 2, 3]
    pred_list = [1, 5, 2, 8, 9] # hits at rank 0 and 2
    
    hr = get_hit_rate(pred_list, ground_truth)
    recall = get_recall(pred_list, ground_truth)
    precision = get_precision(pred_list, ground_truth)
    ndcg = get_ndcg(pred_list, ground_truth)
    
    print(f"User 1 (3 GT, 5 Rec, 2 hit at rank 0, 2):")
    print(f"  HR: {hr} (Expected: 1)")
    print(f"  Recall: {recall:.4f} (Expected: 2/3 = 0.6667)")
    print(f"  Precision: {precision:.4f} (Expected: 2/5 = 0.4000)")
    
    # DCG = 1/log2(0+2) + 1/log2(2+2) = 1/1 + 1/2 = 1.5
    # IDCG = 1/log2(0+2) + 1/log2(1+2) + 1/log2(2+2) (Min of GT size 3 and pred size 5 is 3)
    idcg = 1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)
    expected_ndcg = 1.5 / idcg
    print(f"  NDCG: {ndcg:.4f} (Expected: {expected_ndcg:.4f})")
    
    assert hr == 1
    assert abs(recall - 0.6666) < 1e-3
    assert precision == 0.4
    assert abs(ndcg - expected_ndcg) < 1e-4

    # User 2: No hits
    ground_truth2 = [10, 11]
    pred_list2 = [1, 2, 3, 4, 5]
    print(f"\nUser 2 (No hits):")
    print(f"  HR: {get_hit_rate(pred_list2, ground_truth2)} (Expected: 0)")
    print(f"  Recall: {get_recall(pred_list2, ground_truth2)} (Expected: 0.0)")
    
    assert get_hit_rate(pred_list2, ground_truth2) == 0
    assert get_recall(pred_list2, ground_truth2) == 0.0

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_metrics()
