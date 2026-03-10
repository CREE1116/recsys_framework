import pandas as pd
import numpy as np
import os

def filter_k_core(df, k):
    while True:
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]
        if n_users == df['user_id'].nunique() and n_items == df['item_id'].nunique():
            break
    return df

def create_sampled_subset(input_path, output_path, k, target_items, is_yelp=False):
    print(f"--- Processing {os.path.basename(input_path)} ---")
    if is_yelp:
        df_raw = pd.read_csv(input_path, sep='\t', skiprows=1, header=None, 
                        names=['user_id', 'item_id', 'rating', 'timestamp', 'useful', 'funny', 'cool', 'review_id'])
    else:
        df_raw = pd.read_csv(input_path, sep='\t', header=None, names=['user_id', 'timestamp', 'lat', 'lon', 'item_id'])
    
    # [Fix] Deduplicate User-Item pairs before processing
    print(f"Original interactions: {len(df_raw)}")
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.sort_values('timestamp').drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    else:
        df_raw = df_raw.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    print(f"Unique User-Item interactions: {len(df_raw)}")

    df_core = filter_k_core(df_raw, k)
    unique_users = df_core['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    # Start with a decent guess (e.g., 20% of users or enough to hit target items)
    # Target 35k items. If 50k users give 121k items, try ~15k users.
    n_total_users = len(unique_users)
    n_total_items = df_core['item_id'].nunique()
    
    # Greedy addition
    step = n_total_users // 20 # 5% steps
    current_idx = 0
    final_df = pd.DataFrame()
    
    print(f"Iteratively sampling users from {n_total_users} to hit ~{target_items} items (post k={k} core)...")
    
    while current_idx < n_total_users:
        current_idx += step
        sampled_users = unique_users[:current_idx]
        temp_df = df_core[df_core['user_id'].isin(sampled_users)]
        temp_df = filter_k_core(temp_df, k)
        
        n_items = temp_df['item_id'].nunique()
        print(f"  Users: {len(sampled_users)} -> Final Items: {n_items}")
        
        if n_items >= target_items:
            final_df = temp_df
            break
        
        if current_idx >= n_total_users:
            final_df = temp_df
            break

    print(f"Final Subset: {len(final_df)} interactions, {final_df['user_id'].nunique()} users, {final_df['item_id'].nunique()} items")
    mem_gb = (final_df['item_id'].nunique()**2 * 4) / (1024**3)
    print(f"EASE Memory: {mem_gb:.2f} GB")
    
    final_df.to_csv(output_path, sep='\t', header=is_yelp, index=False)
    print(f"Saved to {output_path}.")
    
    # Save statistics
    stats = {
        "dataset": os.path.basename(output_path),
        "n_interactions": len(final_df),
        "n_users": final_df['user_id'].nunique(),
        "n_items": final_df['item_id'].nunique(),
        "density": len(final_df) / (final_df['user_id'].nunique() * final_df['item_id'].nunique()),
        "avg_interactions_per_user": len(final_df) / final_df['user_id'].nunique(),
        "avg_interactions_per_item": len(final_df) / final_df['item_id'].nunique(),
        "ease_mem_gb": mem_gb
    }
    
    import json
    stats_path = os.path.join(os.path.dirname(output_path), os.path.basename(output_path).replace('.txt', '_stats.json'))
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}.\n")

if __name__ == "__main__":
    # Target ~33,000 items (approx 4.0GB memory) for extra safety.
    TARGET = 33000
    K_CORE = 10
    
    create_sampled_subset("/Users/leejongmin/code/recsys_framework/data/gowalla/loc-gowalla_totalCheckins.txt", 
                         "/Users/leejongmin/code/recsys_framework/data/gowalla/gowalla_subset.txt", K_CORE, TARGET)
    create_sampled_subset("/Users/leejongmin/code/recsys_framework/data/yelp2018/yelp2018.inter", 
                         "/Users/leejongmin/code/recsys_framework/data/yelp2018/yelp2018_subset.txt", K_CORE, TARGET, True)
