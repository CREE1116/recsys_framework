"""
데이터 전처리 유틸리티 모듈.

DataLoader에서 분리된 순수 함수들:
- 데이터 로딩 (load_raw_data, parse_lightgcn_file)
- 필터링 (filter_interactions)
- ID 리매핑 (remap_ids)
- 데이터 분할 (split_leave_one_out, split_temporal_ratio, split_random)
"""
import numpy as np
import pandas as pd


# ============================================================
# 데이터 로딩
# ============================================================

def load_raw_data(data_path, separator, columns, has_header=False):
    """CSV/TSV 등 원시 데이터 파일을 DataFrame으로 로드."""
    engine = 'c'
    if len(separator) > 1 and separator != r'\s+':
        engine = 'python'
    kwargs = {'sep': separator, 'engine': engine}
    if engine == 'c':
        kwargs['low_memory'] = False

    if has_header:
        df = pd.read_csv(data_path, header=0, **kwargs)
        df.columns = columns[:len(df.columns)]
    else:
        df = pd.read_csv(data_path, header=None, names=columns, **kwargs)
    return df


def parse_lightgcn_file(filepath):
    """LightGCN 형식 (user_id item1 item2 ...) 파일을 DataFrame으로 파싱."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            for item_id in parts[1:]:
                rows.append({'user_id': user_id, 'item_id': int(item_id)})
    return pd.DataFrame(rows, dtype=np.int64)


# ============================================================
# 필터링 & 전처리
# ============================================================

def filter_interactions(df, min_user_interactions, min_item_interactions):
    """K-core 필터링: 최소 인터랙션 수를 만족하지 않는 유저/아이템 반복 제거."""
    curr_len = len(df)
    while True:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]

        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]

        new_len = len(df)
        if new_len == curr_len:
            break
        curr_len = new_len
    return df


def dedup_interactions(df, has_timestamp=True):
    """중복 인터랙션 제거 (마지막 인터랙션 유지)."""
    before = len(df)
    if has_timestamp and 'timestamp' in df.columns:
        df = df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'item_id'], keep='last'
        )
    else:
        df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    print(f"[Processing] After dedup: {before} -> {len(df)}")
    return df


def remap_ids(df):
    """유저/아이템 ID를 0-based 연속 정수로 리매핑.
    
    Returns:
        df, user_map, item_map, n_users, n_items, item_popularity
    """
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())

    if not unique_users or not unique_items:
        print("[Processing] WARNING: No users or items found after filtering!")
        df = df.copy()
        df['user_id'] = pd.Series([], dtype=np.int64)
        df['item_id'] = pd.Series([], dtype=np.int64)
        return df, {}, {}, 0, 0, np.array([])

    user_map = {old: new for new, old in enumerate(unique_users)}
    item_map = {old: new for new, old in enumerate(unique_items)}

    df = df.copy()
    df['user_id'] = pd.Categorical(
        df['user_id'], categories=unique_users
    ).codes.astype(np.int64)
    df['item_id'] = pd.Categorical(
        df['item_id'], categories=unique_items
    ).codes.astype(np.int64)

    n_users = len(unique_users)
    n_items = len(unique_items)

    item_popularity = (
        df['item_id'].value_counts()
        .reindex(range(n_items), fill_value=0)
        .sort_index()
        .values
    )

    return df, user_map, item_map, n_users, n_items, item_popularity


# ============================================================
# 데이터 분할
# ============================================================

def split_leave_one_out(df):
    """Leave-One-Out 3-Way Split (Vectorized)."""
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    if 'timestamp' in df.columns:
        df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id'])
    else:
        df_sorted = df.sort_values(by=['user_id', 'item_id'])

    df_sorted = df_sorted.copy()
    df_sorted['_rank'] = (
        df_sorted.groupby('user_id', sort=False)
        .cumcount(ascending=False)
        .astype(np.int64)
    )

    test_df = df_sorted[df_sorted['_rank'] == 0].drop(columns=['_rank'])
    valid_df = df_sorted[df_sorted['_rank'] == 1].drop(columns=['_rank'])
    train_df = df_sorted[df_sorted['_rank'] >= 2].drop(columns=['_rank'])

    return train_df, valid_df, test_df


def split_temporal_ratio(df, train_ratio=0.8, valid_ratio=0.1):
    """User-level Temporal Ratio Split (Vectorized)."""
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id']).copy()

    cum_counts = df_sorted.groupby('user_id', sort=False).cumcount().values
    total_counts = df_sorted['user_id'].map(
        df_sorted.groupby('user_id', sort=False).size()
    ).values.astype(np.int64)

    train_end = np.clip((total_counts * train_ratio).astype(int), 1, None)
    valid_end = (total_counts * (train_ratio + valid_ratio)).astype(int)

    mask_lt3 = total_counts < 3
    valid_end = np.where((total_counts >= 3) & (valid_end >= total_counts), total_counts - 1, valid_end)
    train_end = np.where((total_counts >= 3) & (train_end >= valid_end), valid_end - 1, train_end)

    train_mask = (cum_counts < train_end) | mask_lt3
    valid_mask = (~mask_lt3) & (cum_counts >= train_end) & (cum_counts < valid_end)
    test_mask = (~mask_lt3) & (cum_counts >= valid_end)

    return df_sorted[train_mask].copy(), df_sorted[valid_mask].copy(), df_sorted[test_mask].copy()


def split_random(df, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """Random Ratio Split (Vectorized)."""
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_shuffled = df_shuffled.sort_values(by='user_id', kind='stable').copy()

    cum_counts = df_shuffled.groupby('user_id', sort=False).cumcount().values
    total_counts = df_shuffled['user_id'].map(
        df_shuffled.groupby('user_id', sort=False).size()
    ).values.astype(np.int64)

    train_end = np.clip((total_counts * train_ratio).astype(int), 1, None)
    valid_end = (total_counts * (train_ratio + valid_ratio)).astype(int)

    mask_lt3 = total_counts < 3
    valid_end = np.where((total_counts >= 3) & (valid_end >= total_counts), total_counts - 1, valid_end)
    train_end = np.where((total_counts >= 3) & (train_end >= valid_end), valid_end - 1, train_end)

    train_mask = (cum_counts < train_end) | mask_lt3
    valid_mask = (~mask_lt3) & (cum_counts >= train_end) & (cum_counts < valid_end)
    test_mask = (~mask_lt3) & (cum_counts >= valid_end)

    return df_shuffled[train_mask].copy(), df_shuffled[valid_mask].copy(), df_shuffled[test_mask].copy()


def build_history_dicts(df, train_df, valid_df):
    """유저별 인터랙션 히스토리 딕셔너리 생성.
    
    Returns:
        user_history, train_user_history, eval_user_history
    """
    user_history = df.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
    train_user_history = train_df.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
    eval_df = pd.concat([train_df, valid_df], ignore_index=True)
    eval_user_history = eval_df.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
    return user_history, train_user_history, eval_user_history
