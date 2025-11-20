import os
import pickle
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, TensorDataset
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class RecSysDataset(Dataset):
    """학습을 위한 PyTorch Dataset."""
    def __init__(self, df, n_items, user_history, loss_type, num_negatives, sampling_weights=None):
        self.df = df
        self.n_items = n_items
        self.user_history = user_history
        self.loss_type = loss_type
        self.num_negatives = num_negatives
        self.sampling_weights = sampling_weights

        if self.loss_type == 'pointwise':
            self.users = torch.LongTensor(df['user_id'].values)
            self.items = torch.LongTensor(df['item_id'].values)
            self.ratings = torch.FloatTensor(df['rating'].values)
        elif self.loss_type == 'pairwise':
            self.users = df['user_id'].values
            self.pos_items = df['item_id'].values
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.loss_type == 'pointwise':
            return {
                'user_id': self.users[idx],
                'item_id': self.items[idx],
                'rating': self.ratings[idx]
            }
        elif self.loss_type == 'pairwise':
            user = self.users[idx]
            pos_item = self.pos_items[idx]
            
            # [Optimized] Vectorized Negative Sampling
            neg_items = []
            user_seen = self.user_history[user]
            
            # 1. Sampling from weights (if applicable)
            if self.sampling_weights is not None:
                # Sample more than needed to account for collisions
                num_candidates = self.num_negatives * 2
                candidates = torch.multinomial(self.sampling_weights, num_candidates, replacement=True).tolist()
                
                for cand in candidates:
                    if cand not in user_seen:
                        neg_items.append(cand)
                        if len(neg_items) == self.num_negatives:
                            break
            
            # 2. Uniform Random Sampling (Fill remaining)
            needed = self.num_negatives - len(neg_items)
            if needed > 0:
                # Try to sample all at once
                while True:
                    # Sample a bit more to be safe
                    candidates = np.random.randint(0, self.n_items, size=needed * 2)
                    for cand in candidates:
                        if cand not in user_seen:
                            neg_items.append(cand)
                            if len(neg_items) == self.num_negatives:
                                break
                    if len(neg_items) == self.num_negatives:
                        break
                    # If we still need more, loop again (rare case for sparse datasets)
                    needed = self.num_negatives - len(neg_items)

            return {
                'user_id': torch.LongTensor([user]),
                'pos_item_id': torch.LongTensor([pos_item]),
                'neg_item_id': torch.LongTensor(neg_items)
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

class DataLoader:
    """
    데이터를 로드, 전처리, 분할하는 클래스.
    """
    def __init__(self, config):
        self.config = config
        self.data_path = config['data_path']
        self.separator = config['separator']
        self.columns = config['columns']
        
        self.rating_threshold = config.get('rating_threshold')
        self.min_user_interactions = config['min_user_interactions']
        self.min_item_interactions = config['min_item_interactions']
        
        train_config = config.get('train', {})
        self.loss_type = train_config.get('loss_type', 'pairwise')
        self.num_negatives = train_config.get('num_negatives', 1)
        self.neg_sampling_strategy = train_config.get('negative_sampling_strategy', 'uniform')
        self.neg_sampling_alpha = train_config.get('negative_sampling_alpha', 0.75)

        self.data_cache_path = config.get('data_cache_path', './data_cache/')
        os.makedirs(self.data_cache_path, exist_ok=True)
        
        device = self.config.get('device', 'cpu')
        cache_file_name = f"{config['dataset_name']}_{device}_processed_data.pkl"
        self.cache_file = os.path.join(self.data_cache_path, cache_file_name)

        if os.path.exists(self.cache_file):
            print(f"Loading processed data from cache: {self.cache_file}")
            self._load_from_cache()
        else:
            print("Processing data and saving to cache...")
            self._process_data()
            self._save_to_cache()
            self.test_uni99_negatives = None

        self.sampling_weights = None
        if self.neg_sampling_strategy == 'popularity':
            print(f"Calculating popularity-based sampling weights (alpha={self.neg_sampling_alpha})...")
            counts = torch.FloatTensor(self.item_popularity.values)
            self.sampling_weights = torch.pow(counts, self.neg_sampling_alpha)
            self.sampling_weights /= self.sampling_weights.sum()
            
        if self.config['evaluation'].get('validation_method') == 'uni99' or self.config['evaluation'].get('final_method') == 'uni99':
            if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None:
                print("Loaded pre-sampled test negatives from cache.")
            else:
                self._pre_sample_test_negatives()
                self._save_to_cache()

        print("Data loading and preprocessing complete.")
        print(f"Number of users: {self.n_users}, items: {self.n_items}")
        print(f"Train interactions: {len(self.train_df)}, Test interactions: {len(self.test_df)}")

    def _process_data(self):
        self.df = self._load_data()
        if self.loss_type == 'pairwise' and self.rating_threshold is not None:
            pass
        self.df = self._filter_interactions(self.df)
        self._remap_ids()
        self.train_df, self.test_df = self._split_leave_one_out(self.df)
        self.user_history = self.df.groupby('user_id')['item_id'].apply(set).to_dict()
        # self.interaction_graph = self.get_interaction_graph()

    def _save_to_cache(self):
        data_to_cache = {
            'df': self.df, 'user_map': self.user_map, 'item_map': self.item_map,
            'n_users': self.n_users, 'n_items': self.n_items,
            'train_df': self.train_df, 'test_df': self.test_df,
            'user_history': self.user_history, 'item_popularity': self.item_popularity,
            # 'interaction_graph': self.interaction_graph
        }
        if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None:
            data_to_cache['test_uni99_negatives'] = self.test_uni99_negatives
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data_to_cache, f)

    def _load_from_cache(self):
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            for key, value in cached_data.items():
                setattr(self, key, value)

    def _load_data(self):
        return pd.read_csv(self.data_path, sep=self.separator, header=None, names=self.columns)

    def _filter_interactions(self, df):
        while True:
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            if user_counts.min() >= self.min_user_interactions and item_counts.min() >= self.min_item_interactions:
                break
            df = df[df['user_id'].isin(user_counts[user_counts >= self.min_user_interactions].index)]
            df = df[df['item_id'].isin(item_counts[item_counts >= self.min_item_interactions].index)]
        return df

    def _remap_ids(self):
        self.user_map = {old_id: new_id for new_id, old_id in enumerate(self.df['user_id'].unique())}
        self.item_map = {old_id: new_id for new_id, old_id in enumerate(self.df['item_id'].unique())}
        self.df['user_id'] = self.df['user_id'].map(self.user_map)
        self.df['item_id'] = self.df['item_id'].map(self.item_map)
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        self.item_popularity = self.df['item_id'].value_counts().sort_index().reindex(range(self.n_items), fill_value=0)

    def _split_leave_one_out(self, df):
        df_sorted = df.sort_values(by=['user_id', 'timestamp'])
        test_df = df_sorted.groupby('user_id').tail(1)
        train_df = df.merge(test_df, on=['user_id', 'item_id', 'rating', 'timestamp'], how='left', indicator=True)
        train_df = train_df[train_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        return train_df, test_df

    def _pre_sample_test_negatives(self):
        print("Pre-sampling negative items for uni99 evaluation...")
        self.test_uni99_negatives = {}
        test_user_item_pairs = self.test_df[['user_id', 'item_id']].to_dict('records')

        for row in tqdm(test_user_item_pairs, desc="Sampling uni99 negatives"):
            user_id, pos_item_id = row['user_id'], row['item_id']

            # BUG FIX: user_history의 set을 직접 쓰지 말고, 복사해서 사용
            seen_items = set(self.user_history.get(user_id, set()))
            seen_items.add(pos_item_id)

            negative_items = []
            num_candidates = 200

            while len(negative_items) < 99:
                candidates = np.random.randint(0, self.n_items, size=num_candidates)
                # BUG FIX: set 연산은 복사한 seen_items 기준으로만 수행
                valid_negatives = list(set(candidates) - seen_items)
                negative_items.extend(valid_negatives)
                seen_items.update(valid_negatives)

            self.test_uni99_negatives[user_id] = negative_items[:99]

        print("Pre-sampling complete.")


    def get_interaction_graph(self):
        use_sparse = self.config.get('device') != 'mps'
        num_nodes = self.n_users + self.n_items
        user_ids = torch.LongTensor(self.train_df['user_id'].values)
        item_ids = torch.LongTensor(self.train_df['item_id'].values)
        if use_sparse:
            row_indices = torch.cat([user_ids, item_ids + self.n_users])
            col_indices = torch.cat([item_ids + self.n_users, user_ids])
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
            eye_indices = torch.arange(num_nodes)
            adj_matrix[eye_indices, eye_indices] = 1.0
            return adj_matrix

    def get_train_loader(self, batch_size):
        train_dataset = RecSysDataset(
            self.train_df, self.n_items, self.user_history,
            self.loss_type, self.num_negatives, self.sampling_weights
        )
        use_pin_memory = self.config.get('device', 'cpu') != 'cpu'
        return PyTorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            pin_memory=use_pin_memory
        )

    def get_test_df(self):
        return self.test_df
        
    def get_test_uni99_negatives(self):
        return self.test_uni99_negatives

    def get_uni99_test_loader(self, batch_size):
        if self.test_uni99_negatives is None:
            raise ValueError("Pre-sampled negatives for uni99 not found. Check config and data_loader setup.")
        test_dataset = Uni99RecSysDataset(self.test_df, self.test_uni99_negatives)
        use_pin_memory = self.config.get('device', 'cpu') != 'cpu'
        return PyTorchDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            pin_memory=use_pin_memory
        )

    def get_full_test_loader(self, batch_size):
        test_dataset = TensorDataset(
            torch.LongTensor(self.test_df['user_id'].values),
            torch.LongTensor(self.test_df['item_id'].values)
        )
        return PyTorchDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )

    def get_validation_loader(self, batch_size):
        """[추가] 설정에 맞는 검증용 데이터 로더를 반환합니다."""
        method = self.config['evaluation'].get('validation_method', 'uni99')
        if method == 'uni99':
            return self.get_uni99_test_loader(batch_size)
        elif method == 'full':
            return self.get_full_test_loader(batch_size)
        else:
            raise ValueError(f"Unsupported validation method: {method}")

    def get_final_loader(self, batch_size):
        """[추가] 설정에 맞는 최종 평가용 데이터 로더를 반환합니다."""
        method = self.config['evaluation'].get('final_method', 'full')
        if method == 'uni99':
            return self.get_uni99_test_loader(batch_size)
        elif method == 'full':
            return self.get_full_test_loader(batch_size)
        else:
            raise ValueError(f"Unsupported final method: {method}")
