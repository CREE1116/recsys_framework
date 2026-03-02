import pandas as pd
import os

def check_k_core(data_path, k_list, is_yelp=False):
    print(f"--- Loading {os.path.basename(data_path)} ---")
    if is_yelp:
        df_raw = pd.read_csv(data_path, sep='\t', skiprows=1, header=None, 
                        names=['user_id', 'item_id', 'rating', 'timestamp', 'useful', 'funny', 'cool', 'review_id'])
    else:
        df_raw = pd.read_csv(data_path, sep='\t', header=None, names=['user_id', 'timestamp', 'lat', 'lon', 'item_id'])
    
    print(f"Raw: {len(df_raw)} interactions, {df_raw['user_id'].nunique()} users, {df_raw['item_id'].nunique()} items")
    
    for k in k_list:
        df = df_raw.copy()
        while True:
            n_users = df['user_id'].nunique()
            n_items = df['item_id'].nunique()
            
            user_counts = df['user_id'].value_counts()
            df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
            
            item_counts = df['item_id'].value_counts()
            df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]
            
            if n_users == df['user_id'].nunique() and n_items == df['item_id'].nunique():
                break
        
        if len(df) == 0:
            print(f"k={k:2d}: Empty")
            continue
            
        mem_gb = (df['item_id'].nunique()**2 * 4) / (1024**3)
        density = len(df) / (df['user_id'].nunique() * df['item_id'].nunique())
        print(f"k={k:2d}: {len(df):7d} interactions, {df['user_id'].nunique():6d} users, {df['item_id'].nunique():6d} items. Density: {density:.6f}, EASE Mem: {mem_gb:6.2f} GB")

if __name__ == "__main__":
    check_k_core("/Users/leejongmin/code/recsys_framework/data/gowalla/loc-gowalla_totalCheckins.txt", [5, 10])
    check_k_core("/Users/leejongmin/code/recsys_framework/data/yelp2018/yelp2018.inter", [5, 10], is_yelp=True)
