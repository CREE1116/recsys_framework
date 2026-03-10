import os
import sys
import pickle
import time
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, TensorDataset
import torch
import pandas as pd
import numpy as np

from .data_processing import (
    load_raw_data, parse_lightgcn_file,
    filter_interactions, dedup_interactions,
    remap_ids, split_leave_one_out, split_temporal_ratio, split_random,
    build_history_dicts,
)


# ============================================================
# Dataset 클래스
# ============================================================

class RecSysDataset(Dataset):
    """학습을 위한 PyTorch Dataset."""
    def __init__(self, df, n_items, user_history, loss_type, num_negatives, sampling_weights=None, train_user_history=None):
        self.df = df
        self.n_items = n_items
        self.user_history = user_history
        self.loss_type = loss_type
        self.num_negatives = num_negatives
        self.sampling_weights = sampling_weights
        self.train_user_history = train_user_history if train_user_history is not None else user_history

        self.users = df['user_id'].values
        self.items = df['item_id'].values
        if self.loss_type == 'pointwise':
            self.ratings = df['rating'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.loss_type == 'pointwise':
            return self.users[idx], self.items[idx], self.ratings[idx]
        else:
            return self.users[idx], self.items[idx]

    def collate_fn(self, batch):
        if self.loss_type == 'pointwise':
            users, items, ratings = zip(*batch)
            return {
                'user_id': torch.LongTensor(users),
                'item_id': torch.LongTensor(items),
                'rating': torch.FloatTensor(ratings)
            }
        else:
            users, items = zip(*batch)
            users_np = np.array(users)
            items_np = np.array(items)
            B = len(users_np)
            N = self.num_negatives

            # Vectorized negative sampling
            all_neg = np.random.randint(0, self.n_items, size=(B, N * 3))
            
            if self.sampling_weights is not None:
                weights_np = self.sampling_weights.numpy()
                all_neg = np.random.choice(
                    self.n_items, size=(B, N * 3), p=weights_np
                )

            final_neg = np.zeros((B, N), dtype=np.int64)
            for i in range(B):
                seen = self.train_user_history.get(int(users_np[i]), set())
                pos_item = int(items_np[i])
                candidates = all_neg[i]
                # O(N*3) set lookup instead of O(|seen| × N*3) numpy comparison
                valid = [c for c in candidates if c != pos_item and c not in seen]
                if len(valid) >= N:
                    final_neg[i] = valid[:N]
                else:
                    final_neg[i, :len(valid)] = valid
                    remaining = N - len(valid)
                    while remaining > 0:
                        s = np.random.randint(0, self.n_items)
                        if s != pos_item and s not in seen:
                            final_neg[i, N - remaining] = s
                            remaining -= 1

            return {
                'user_id': torch.LongTensor(users_np).unsqueeze(1),
                'pos_item_id': torch.LongTensor(items_np).unsqueeze(1),
                'neg_item_id': torch.LongTensor(final_neg)
            }


class Uni99RecSysDataset(Dataset):
    """uni99 평가 전용 PyTorch Dataset."""
    def __init__(self, test_df, test_uni99_negatives):
        self.users = test_df['user_id'].values
        self.pos_items = test_df['item_id'].values
        self.neg_items = [test_uni99_negatives[u] for u in self.users]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_items = self.neg_items[idx]
        items_to_rank = [pos_item] + neg_items
        return {
            'user_id': torch.LongTensor([user]),
            'items': torch.LongTensor(items_to_rank)
        }


# ============================================================
# 로더 팩토리 헬퍼
# ============================================================

def _make_loader(dataset, batch_size, shuffle, num_workers, prefetch_factor, pin_memory, collate_fn=None):
    """[Fix] prefetch_factor/persistent_workers는 num_workers > 0일 때만 설정."""
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if collate_fn:
        kwargs['collate_fn'] = collate_fn
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = True
    return PyTorchDataLoader(**kwargs)


# ============================================================
# DataLoader 메인 클래스
# ============================================================

class DataLoader:
    """
    데이터를 로드, 전처리, 분할하고 모델 학습에 필요한 형태로 제공하는 클래스.
    실제 전처리 로직은 data_processing 모듈에 위임합니다.
    """
    def __init__(self, config):
        self.config = config

        train_config = config.get('train', {})
        self.loss_type = train_config.get('loss_type', 'pairwise')
        self.num_negatives = train_config.get('num_negatives', 1)
        self.neg_sampling_strategy = train_config.get('negative_sampling_strategy', 'uniform')
        self.neg_sampling_alpha = train_config.get('negative_sampling_alpha', 0.75)

        self.data_cache_path = config.get('data_cache_path', './data_cache/')
        os.makedirs(self.data_cache_path, exist_ok=True)

        # 데이터 처리 또는 캐시 로드
        split_method = config.get('split_method', 'loo')
        self._load_or_process_data(split_method)

        # item_popularity 통일 numpy array
        if hasattr(self, 'item_popularity'):
            self.item_popularity = np.asarray(self.item_popularity)

        # 캐시 로드 후 history 누락 복구
        if not hasattr(self, 'train_user_history'):
            print("Recovering train_user_history from train_df...")
            self.train_user_history = (
                self.train_df.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
            )
        if not hasattr(self, 'eval_user_history'):
            print("Recovering eval_user_history from train_df + valid_df...")
            eval_df = pd.concat([self.train_df, self.valid_df], ignore_index=True)
            self.eval_user_history = (
                eval_df.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
            )

        # Popularity 기반 negative sampling weights
        self.sampling_weights = None
        if self.neg_sampling_strategy == 'popularity':
            print(f"Calculating popularity-based sampling weights (alpha={self.neg_sampling_alpha})...")
            if not hasattr(self, 'item_popularity') or self.item_popularity is None or len(self.item_popularity) == 0:
                self.item_popularity = (
                    self.train_df['item_id'].value_counts()
                    .reindex(range(self.n_items), fill_value=0)
                    .sort_index()
                    .values
                )
            counts = torch.FloatTensor(self.item_popularity)
            self.sampling_weights = torch.pow(counts, self.neg_sampling_alpha)
            if self.sampling_weights.sum() > 0:
                self.sampling_weights /= self.sampling_weights.sum()

        # Uni99 네거티브 샘플링 처리
        self._handle_uni99_negatives(split_method)

        print("Data loading and preprocessing complete.")
        print(f"Number of users: {self.n_users}, items: {self.n_items}")
        print(f"Train: {len(self.train_df)}, Valid: {len(self.valid_df)}, Test: {len(self.test_df)}")

    # ----------------------------------------------------------
    # 내부: 데이터 로드/캐시 관리
    # ----------------------------------------------------------

    def _load_or_process_data(self, split_method):
        """캐시가 있으면 로드, 없으면 전처리 후 캐시 저장."""
        if split_method == 'presplit':
            cache_file_name = f"{self.config['dataset_name']}_presplit.pkl"
            self.cache_file = os.path.join(self.data_cache_path, cache_file_name)
            if os.path.exists(self.cache_file):
                print(f"[DataLoader] Cache hit")
                self._load_from_cache()
            else:
                print("[DataLoader] Processing presplit data...")
                self._process_presplit_data()
        else:
            rt = self.config.get('rating_threshold', 'none')
            min_u = self.config.get('min_user_interactions', 5)
            min_i = self.config.get('min_item_interactions', 5)
            tr = self.config.get('train_ratio', 0.8)
            vr = self.config.get('valid_ratio', 0.1)
            dedup = self.config.get('dedup', True)
            cache_file_name = (
                f"{self.config['dataset_name']}_{split_method}"
                f"_rt{rt}_mu{min_u}_mi{min_i}_tr{tr}_vr{vr}_dedup{int(dedup)}.pkl"
            )
            self.cache_file = os.path.join(self.data_cache_path, cache_file_name)

            if os.path.exists(self.cache_file):
                print(f"[DataLoader] Cache hit")
                self._load_from_cache()
            else:
                print("[DataLoader] Processing data...")
                self._process_data()

    def _process_presplit_data(self):
        """Pre-split 데이터 처리 (LightGCN 형식)."""
        train_file = self.config['train_file']
        test_file = self.config['test_file']

        print(f"Loading presplit data...")
        self.train_df = parse_lightgcn_file(train_file)
        self.test_df = parse_lightgcn_file(test_file)
        self.valid_df = pd.DataFrame(columns=['user_id', 'item_id']).astype(np.int64)
        self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        self.n_users = int(self.df['user_id'].max()) + 1 if len(self.df) > 0 else 0
        self.n_items = int(self.df['item_id'].max()) + 1 if len(self.df) > 0 else 0
        self.user_map = {u: u for u in range(self.n_users)}
        self.item_map = {i: i for i in range(self.n_items)}

        self.user_history, self.train_user_history, self.eval_user_history = \
            build_history_dicts(self.df, self.train_df, self.valid_df)
        self.item_popularity = (
            self.train_df['item_id'].value_counts()
            .reindex(range(self.n_items), fill_value=0)
            .sort_index()
            .values
        )
        print(f"Presplit data loaded: {self.n_users} users, {self.n_items} items")

    def _process_data(self):
        """일반 데이터 전처리: 로드 → 필터 → 리매핑 → 분할."""
        start_time = time.time()
        
        df = load_raw_data(
            self.config['data_path'],
            self.config['separator'],
            self.config['columns'],
            self.config.get('has_header', False)
        )
        print(f"[DataLoader] Raw data loaded: {len(df)} lines ({time.time() - start_time:.2f}s)")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)

        # Rating threshold 필터링
        if self.config.get('rating_threshold') is not None:
            rating_col = self.config['columns'][2]
            if rating_col in df.columns:
                df = df[pd.to_numeric(df[rating_col], errors='coerce').fillna(0) >= self.config['rating_threshold']]
            print(f"[DataLoader] After threshold filtering: {len(df)}")

        # K-core 필터링
        df = filter_interactions(
            df,
            self.config['min_user_interactions'],
            self.config['min_item_interactions']
        )
        print(f"[DataLoader] After k-core filtering: {len(df)}")

        # 중복 제거
        if self.config.get('dedup', True):
            df = dedup_interactions(df, 'timestamp' in df.columns)

        # ID 리매핑
        df, self.user_map, self.item_map, self.n_users, self.n_items, self.item_popularity = remap_ids(df)
        self.df = df
        print(f"[DataLoader] ID Remapping complete.")

        # 분할
        split_method = self.config.get('split_method', 'loo')
        if split_method == 'loo':
            self.train_df, self.valid_df, self.test_df = split_leave_one_out(df)
        elif split_method in ('temporal_ratio', 'temporal'):
            self.train_df, self.valid_df, self.test_df = split_temporal_ratio(
                df, self.config.get('train_ratio', 0.8), self.config.get('valid_ratio', 0.1)
            )
        elif split_method == 'random':
            self.train_df, self.valid_df, self.test_df = split_random(
                df, self.config.get('train_ratio', 0.8), self.config.get('valid_ratio', 0.1),
                self.config.get('seed', 42)
            )
        else:
            raise ValueError(f"Unknown split_method: {split_method}.")
        print(f"[DataLoader] Split ({split_method}) complete.")

        # 히스토리 생성
        self.user_history, self.train_user_history, self.eval_user_history = \
            build_history_dicts(df, self.train_df, self.valid_df)
        print(f"[DataLoader] Total processing time: {time.time() - start_time:.2f}s")

        # 캐시 저장
        self._save_to_cache()

    def _save_to_cache(self):
        data_to_cache = {
            'df': self.df,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'train_df': self.train_df,
            'valid_df': self.valid_df,
            'test_df': self.test_df,
            'user_history': getattr(self, 'user_history', {}),
            'train_user_history': getattr(self, 'train_user_history', {}),
            'eval_user_history': getattr(self, 'eval_user_history', {}),
            'item_popularity': getattr(self, 'item_popularity', np.array([])),
        }
        if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None:
            data_to_cache['test_uni99_negatives'] = self.test_uni99_negatives
        if hasattr(self, 'valid_uni99_negatives') and self.valid_uni99_negatives is not None:
            data_to_cache['valid_uni99_negatives'] = self.valid_uni99_negatives
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data_to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[DataLoader] Cache saved")

    def _load_from_cache(self):
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        for key, value in cached_data.items():
            setattr(self, key, value)

    def _handle_uni99_negatives(self, split_method):
        """Uni99 평가 방식에 필요한 네거티브 샘플 처리."""
        eval_config = self.config.get('evaluation', {})
        need_uni99 = (eval_config.get('validation_method') == 'uni99' or
                      eval_config.get('final_method') == 'uni99')

        uni99_loaded = (
            hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None and
            hasattr(self, 'valid_uni99_negatives') and self.valid_uni99_negatives is not None
        )

        if need_uni99 and uni99_loaded:
            print("Loaded pre-sampled uni99 negatives (valid & test) from cache.")
        elif need_uni99 and not uni99_loaded:
            self._pre_sample_test_negatives()
            self._save_to_cache()
        elif not os.path.exists(self.cache_file):
            self._save_to_cache()

    def _pre_sample_test_negatives(self):
        """Vectorized Negative Sampling for uni99 evaluation."""
        print("[DataLoader] Pre-sampling negative items for uni99 evaluation...")

        for name, split_df, history in [
            ('test', self.test_df, self.eval_user_history),
            ('valid', self.valid_df, self.train_user_history)
        ]:
            if split_df.empty:
                if name == 'test':
                    self.test_uni99_negatives = {}
                else:
                    self.valid_uni99_negatives = {}
                continue

            user_ids = split_df['user_id'].values
            pos_item_ids = split_df['item_id'].values
            negatives_dict = {}

            chunk_size = 10000
            for i in range(0, len(user_ids), chunk_size):
                chunk_users = user_ids[i:i + chunk_size]
                chunk_pos = pos_item_ids[i:i + chunk_size]
                samples = np.random.randint(0, max(1, self.n_items), size=(len(chunk_users), 300))

                for j, user_id in enumerate(chunk_users):
                    pos_item = int(chunk_pos[j])
                    seen = history.get(int(user_id), set())
                    row_samples = samples[j]

                    finals_set = set()
                    finals_list = []
                    for s in row_samples:
                        s = int(s)
                        if s != pos_item and s not in seen and s not in finals_set:
                            finals_set.add(s)
                            finals_list.append(s)
                            if len(finals_list) == 99:
                                break

                    if len(finals_list) < 99 and self.n_items > 0:
                        max_possible = self.n_items - len(seen) - 1
                        target = min(99, max_possible)
                        while len(finals_list) < target:
                            s = int(np.random.randint(0, self.n_items))
                            if s != pos_item and s not in seen and s not in finals_set:
                                finals_set.add(s)
                                finals_list.append(s)

                    negatives_dict[int(user_id)] = finals_list

            if name == 'test':
                self.test_uni99_negatives = negatives_dict
            else:
                self.valid_uni99_negatives = negatives_dict

        print("[DataLoader] Pre-sampling complete.")

    # ----------------------------------------------------------
    # 그래프 구성
    # ----------------------------------------------------------

    def get_interaction_graph(self, add_self_loops=False):
        device = self.config.get('device', 'cpu')
        use_sparse = device not in ('mps',)

        num_nodes = self.n_users + self.n_items
        if num_nodes == 0:
            if use_sparse:
                return torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (0, 0))
            return torch.zeros((0, 0))

        user_ids = torch.LongTensor(self.train_df['user_id'].values)
        item_ids = torch.LongTensor(self.train_df['item_id'].values)

        if use_sparse:
            row_indices = torch.cat([user_ids, item_ids + self.n_users])
            col_indices = torch.cat([item_ids + self.n_users, user_ids])

            if add_self_loops:
                self_loop_indices = torch.arange(num_nodes)
                row_indices = torch.cat([row_indices, self_loop_indices])
                col_indices = torch.cat([col_indices, self_loop_indices])

            all_values = torch.ones(len(row_indices))
            indices = torch.stack([row_indices, col_indices])
            return torch.sparse_coo_tensor(indices, all_values, (num_nodes, num_nodes))
        else:
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
            item_ids_offset = item_ids + self.n_users
            adj_matrix[user_ids, item_ids_offset] = 1.0
            adj_matrix[item_ids_offset, user_ids] = 1.0
            if add_self_loops:
                eye_indices = torch.arange(num_nodes)
                adj_matrix[eye_indices, eye_indices] = 1.0
            return adj_matrix

    # ----------------------------------------------------------
    # 로더 팩토리
    # ----------------------------------------------------------

    def _get_loader_kwargs(self):
        device = self.config.get('device', 'cpu')
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        num_workers = self.config.get('num_workers', 4)
        if sys.platform == 'darwin':
            num_workers = 0
        prefetch_factor = self.config.get('prefetch_factor', 2)
        pin_memory = device not in ('cpu', 'mps')
        return num_workers, prefetch_factor, pin_memory

    def get_train_loader(self, batch_size):
        train_dataset = RecSysDataset(
            self.train_df, self.n_items,
            self.train_user_history, self.loss_type,
            self.num_negatives, self.sampling_weights,
            train_user_history=self.train_user_history
        )
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(train_dataset, batch_size, shuffle=True,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, collate_fn=train_dataset.collate_fn)

    def get_test_df(self):
        return self.test_df

    def get_test_uni99_negatives(self):
        return getattr(self, 'test_uni99_negatives', {})

    def get_uni99_test_loader(self, batch_size):
        uni99_dataset = Uni99RecSysDataset(self.test_df, self.test_uni99_negatives)
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(uni99_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    def get_uni99_valid_loader(self, batch_size):
        uni99_dataset = Uni99RecSysDataset(self.valid_df, self.valid_uni99_negatives)
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(uni99_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    def get_full_test_loader(self, batch_size):
        test_dataset = TensorDataset(
            torch.LongTensor(self.test_df['user_id'].values),
            torch.LongTensor(self.test_df['item_id'].values)
        )
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(test_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    def get_sampled_full_test_loader(self, batch_size, ratio=0.2):
        unique_users = self.test_df['user_id'].unique()
        num_sample = max(1, int(len(unique_users) * ratio))
        rng = np.random.RandomState(42)
        sampled_users = rng.choice(unique_users, num_sample, replace=False)
        sampled_df = self.test_df[self.test_df['user_id'].isin(sampled_users)].copy()
        test_dataset = TensorDataset(
            torch.LongTensor(sampled_df['user_id'].values),
            torch.LongTensor(sampled_df['item_id'].values)
        )
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(test_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    @property
    def has_validation(self):
        return hasattr(self, 'valid_df') and len(self.valid_df) > 0

    def get_full_valid_loader(self, batch_size):
        if not self.has_validation:
            return self.get_full_test_loader(batch_size)
        valid_dataset = TensorDataset(
            torch.LongTensor(self.valid_df['user_id'].values),
            torch.LongTensor(self.valid_df['item_id'].values)
        )
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(valid_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    def get_sampled_valid_loader(self, batch_size, ratio=0.2):
        if not self.has_validation:
            return self.get_sampled_full_test_loader(batch_size, ratio=ratio)
        unique_users = self.valid_df['user_id'].unique()
        num_sample = max(1, int(len(unique_users) * ratio))
        rng = np.random.RandomState(42)
        sampled_users = rng.choice(unique_users, num_sample, replace=False)
        sampled_df = self.valid_df[self.valid_df['user_id'].isin(sampled_users)].copy()
        valid_dataset = TensorDataset(
            torch.LongTensor(sampled_df['user_id'].values),
            torch.LongTensor(sampled_df['item_id'].values)
        )
        num_workers, prefetch_factor, pin_memory = self._get_loader_kwargs()
        return _make_loader(valid_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory)

    def get_validation_loader(self, batch_size):
        method = self.config['evaluation'].get('validation_method', 'full')
        if method == 'full':
            return self.get_full_valid_loader(batch_size)
        elif method == 'sampled':
            ratio = self.config['evaluation'].get('validation_sample_ratio', 0.2)
            return self.get_sampled_valid_loader(batch_size, ratio=ratio)
        elif method == 'uni99':
            return self.get_uni99_valid_loader(batch_size)
        else:
            raise ValueError(f"Unsupported validation method: {method}.")

    def get_final_loader(self, batch_size):
        method = self.config['evaluation'].get('final_method', 'full')
        if method == 'full':
            return self.get_full_test_loader(batch_size)
        elif method == 'uni99':
            return self.get_uni99_test_loader(batch_size)
        else:
            raise ValueError(f"Unsupported final method: {method}.")