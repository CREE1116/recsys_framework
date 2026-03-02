import numpy as np
import pandas as pd
import torch
from src.evaluation import get_pop_ratio, get_long_tail_item_set, get_novelty, get_long_tail_coverage

def test_metrics_robustness():
    print("--- Testing evaluation metrics robustness ---")
    
    # Test cases for item_popularity types
    n_items = 10
    pop_arr = np.array([10, 5, 20, 15, 30, 2, 8, 12, 18, 25])
    pop_series = pd.Series(pop_arr)
    pop_dict = {i: v for i, v in enumerate(pop_arr)}
    
    mean_pop = np.mean(pop_arr)
    target_item = 4
    
    # 1. Test get_pop_ratio
    print("Testing get_pop_ratio...")
    res_arr = get_pop_ratio(target_item, pop_arr, mean_pop)
    res_series = get_pop_ratio(target_item, pop_series, mean_pop)
    res_dict = get_pop_ratio(target_item, pop_dict, mean_pop)
    assert res_arr == res_series == res_dict
    print("  PASSED")
    
    # 2. Test get_long_tail_item_set
    print("Testing get_long_tail_item_set...")
    set_arr = get_long_tail_item_set(pop_arr, 0.8)
    set_series = get_long_tail_item_set(pop_series, 0.8)
    assert set_arr == set_series
    print("  PASSED")
    
    # 3. Test get_novelty
    print("Testing get_novelty...")
    recs = [0, 4, 8]
    nov_arr = get_novelty(recs, pop_arr)
    nov_series = get_novelty(recs, pop_series)
    nov_dict = get_novelty(recs, pop_dict)
    # They should be very close (floating point)
    assert np.allclose(nov_arr, nov_series)
    assert np.allclose(nov_arr, nov_dict)
    print("  PASSED")
    
    print("--- All metrics robustness tests PASSED ---")

if __name__ == "__main__":
    test_metrics_robustness()
