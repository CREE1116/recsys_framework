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

        # Standardize storage: 'items' generically
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        # Store ratings separately if needed for pointwise loss, as __getitem__ won't return it directly
        if self.loss_type == 'pointwise':
            self.ratings = df['rating'].values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_data = {
            'user_id': self.users[idx],
            'item_id': self.items[idx]
        }
        if self.loss_type == 'pointwise':
            item_data['rating'] = self.ratings[idx]
        return item_data

    def collate_fn(self, batch):
        if self.loss_type == 'pointwise':
            # No Negative Sampling - Just return batch as is
            return torch.utils.data.default_collate(batch)
        
        # Pairwise: Perform Negative Sampling and Rename item_id -> pos_item_id
        users = [item['user_id'] for item in batch]
        pos_items = [item['item_id'] for item in batch]
        
        batch_size = len(batch)
        users_tensor = torch.tensor(users, dtype=torch.long)
        pos_items_tensor = torch.tensor(pos_items, dtype=torch.long)
        
        neg_items_list = []
        
        # [RecBole 표준] 각 (u, pos) 쌍에 대해 개별적으로 negative sampling
        # 배치가 candidates pool을 공유하지 않음
        for i in range(batch_size):
            u = users[i]
            user_seen = self.user_history[u]
            u_negs = []
            
            while len(u_negs) < self.num_negatives:
                # 각 샘플에 대해 독립적으로 random sampling
                if self.sampling_weights is not None:
                    c = torch.multinomial(self.sampling_weights, 1, replacement=True).item()
                else:
                    c = torch.randint(0, self.n_items, (1,)).item()
                
                if c not in user_seen:
                    u_negs.append(c)
            
            neg_items_list.extend(u_negs)
            
        neg_items_tensor = torch.tensor(neg_items_list, dtype=torch.long)
        
        # Reshape: Always (batch_size, num_negatives) to maintain consistent 2D shape,
        # even if num_negatives=1 or batch_size=1.
        neg_items_tensor = neg_items_tensor.view(batch_size, self.num_negatives)
        
        return {
            'user_id': users_tensor,
            'pos_item_id': pos_items_tensor,
            'neg_item_id': neg_items_tensor
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
    데이터를 로드, 전처리, 분할하고 모델 학습에 필요한 형태로 제공하는 클래스.
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
            # 캐시 로드 시 item_popularity가 없을 경우를 대비해 다시 계산 (일반적으로는 _process_data에서 처리됨)
            if not hasattr(self, 'item_popularity'):
                 self.item_popularity = self.train_df['item_id'].value_counts().sort_index().reindex(range(self.n_items), fill_value=0)

            counts = torch.FloatTensor(self.item_popularity.values)
            self.sampling_weights = torch.pow(counts, self.neg_sampling_alpha)
            self.sampling_weights /= self.sampling_weights.sum()
            
        # 검증 방식이 uni99인 경우 네거티브 샘플링 수행
        if self.config['evaluation'].get('validation_method') == 'uni99' or self.config['evaluation'].get('final_method') == 'uni99':
            if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None:
                print("Loaded pre-sampled test negatives from cache.")
            else:
                self._pre_sample_test_negatives()
                self._save_to_cache()

        if not hasattr(self, 'train_user_history'):
             print("Generating train_user_history from train_df...")
             self.train_user_history = self.train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        
        print("Data loading and preprocessing complete.")
        print(f"Number of users: {self.n_users}, items: {self.n_items}")
        print(f"Train: {len(self.train_df)}, Valid: {len(self.valid_df)}, Test: {len(self.test_df)}")

    def _process_data(self):
        self.df = self._load_data()
        
        # Rating Threshold 필터링 (Pairwise, Listwise, Pointwise 모두 적용)
        if self.rating_threshold is not None:
             # 보통 Rating 컬럼은 3번째 (인덱스 2)에 위치
             rating_col = self.columns[2] 
             if rating_col in self.df.columns:
                self.df = self.df[self.df[rating_col] >= self.rating_threshold]

        self.df = self._filter_interactions(self.df)
        self._remap_ids()
        self.train_df, self.valid_df, self.test_df = self._split_leave_one_out(self.df)
        self.user_history = self.df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.train_user_history = self.train_df.groupby('user_id')['item_id'].apply(set).to_dict()

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
            'user_history': self.user_history, 
            'train_user_history': self.train_user_history,
            'item_popularity': self.item_popularity,
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
        # engine='python'을 사용하여 다양한 구분자 지원 강화
        return pd.read_csv(self.data_path, sep=self.separator, header=None, names=self.columns, engine='python')

    def _filter_interactions(self, df):
        # 최소 상호작용 수를 만족할 때까지 반복 필터링 (k-core filtering)
        while True:
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            
            if user_counts.min() >= self.min_user_interactions and item_counts.min() >= self.min_item_interactions:
                break
            
            df = df[df['user_id'].isin(user_counts[user_counts >= self.min_user_interactions].index)]
            df = df[df['item_id'].isin(item_counts[item_counts >= self.min_item_interactions].index)]
        return df

    def _remap_ids(self):
        # User/Item ID를 0부터 시작하는 연속된 정수로 매핑
        self.user_map = {old_id: new_id for new_id, old_id in enumerate(self.df['user_id'].unique())}
        self.item_map = {old_id: new_id for new_id, old_id in enumerate(self.df['item_id'].unique())}
        
        self.df['user_id'] = self.df['user_id'].map(self.user_map)
        self.df['item_id'] = self.df['item_id'].map(self.item_map)
        
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        
        self.item_popularity = self.df['item_id'].value_counts().sort_index().reindex(range(self.n_items), fill_value=0)

    def _split_leave_one_out(self, df):
        """
        Leave-One-Out 3-Way Split (RecBole 표준)
        - Test: 유저별 마지막 상호작용
        - Valid: 유저별 끝에서 2번째 상호작용
        - Train: 나머지
        """
        # Timestamp 기준 정렬 (동일 timestamp 처리를 위해 item_id를 tie-breaker로 사용)
        df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id'])
        
        # Test: 마지막 아이템
        test_df = df_sorted.groupby('user_id').tail(1)
        
        # Valid: 끝에서 2번째 아이템 (Test 제외 후 마지막)
        remaining_after_test = df_sorted.merge(
            test_df, on=['user_id', 'item_id', 'rating', 'timestamp'], 
            how='left', indicator=True
        )
        remaining_after_test = remaining_after_test[remaining_after_test['_merge'] == 'left_only'].drop(columns=['_merge'])
        valid_df = remaining_after_test.groupby('user_id').tail(1)
        
        # Train: Test, Valid 모두 제외
        eval_df = pd.concat([test_df, valid_df], ignore_index=True)
        train_df = df.merge(
            eval_df, on=['user_id', 'item_id', 'rating', 'timestamp'], 
            how='left', indicator=True
        )
        train_df = train_df[train_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        return train_df, valid_df, test_df

    def _pre_sample_test_negatives(self):
        print("Pre-sampling negative items for uni99 evaluation...")
        self.test_uni99_negatives = {}
        test_user_item_pairs = self.test_df[['user_id', 'item_id']].to_dict('records')

        for row in tqdm(test_user_item_pairs, desc="Sampling uni99 negatives"):
            user_id, pos_item_id = row['user_id'], row['item_id']

            # [BUG FIX] set()을 사용하여 user_history의 복사본을 생성합니다.
            # 원본 self.user_history를 직접 수정하면 다른 에폭이나 로직에 영향을 줍니다.
            seen_items = set(self.user_history.get(user_id, set()))
            seen_items.add(pos_item_id)

            negative_items = []
            num_candidates = 200 # 충돌을 고려하여 넉넉하게 샘플링

            while len(negative_items) < 99:
                candidates = np.random.randint(0, self.n_items, size=num_candidates)
                # [BUG FIX] 복사된 seen_items와 비교
                valid_negatives = list(set(candidates) - seen_items)
                
                negative_items.extend(valid_negatives)
                
                # 중복 방지를 위해 이번 루프에서 찾은 네거티브도 seen에 추가
                seen_items.update(valid_negatives)

            self.test_uni99_negatives[user_id] = negative_items[:99]

        print("Pre-sampling complete.")

    def get_interaction_graph(self, add_self_loops=False):
        """
        LightGCN 등 GNN 모델을 위한 User-Item 이분 그래프(Adjacency Matrix)를 생성합니다.
        """
        # MPS(Mac) 장치는 Sparse Tensor 지원이 제한적이므로 Dense Matrix 사용
        use_sparse = self.config.get('device') != 'mps'
        
        num_nodes = self.n_users + self.n_items
        user_ids = torch.LongTensor(self.train_df['user_id'].values)
        item_ids = torch.LongTensor(self.train_df['item_id'].values)
        
        if use_sparse:
            # 희소 행렬 (Sparse Matrix) 생성
            # 행: [Users, Items], 열: [Items, Users] 형태의 대칭 구조
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
            # 밀집 행렬 (Dense Matrix) 생성
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
            item_ids_offset = item_ids + self.n_users
            
            # User <-> Item 연결
            adj_matrix[user_ids, item_ids_offset] = 1.0
            adj_matrix[item_ids_offset, user_ids] = 1.0
            
            if add_self_loops:
                eye_indices = torch.arange(num_nodes)
                adj_matrix[eye_indices, eye_indices] = 1.0
                
            return adj_matrix

    def get_train_loader(self, batch_size):
        # [BUG FIX] RecBole 표준에 맞게 train_user_history 사용
        # 학습 시점에는 test 데이터를 모르는 것으로 가정
        train_dataset = RecSysDataset(
            self.train_df, self.n_items, self.train_user_history,
            self.loss_type, self.num_negatives, self.sampling_weights
        )
        use_pin_memory = self.config.get('device', 'cpu') != 'cpu'
        
        return PyTorchDataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            pin_memory=use_pin_memory,
            collate_fn=train_dataset.collate_fn
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
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2),
            pin_memory=use_pin_memory
        )

    def get_full_test_loader(self, batch_size):
        """전체 아이템에 대한 랭킹 평가를 위한 로더"""
        test_dataset = TensorDataset(
            torch.LongTensor(self.test_df['user_id'].values),
            torch.LongTensor(self.test_df['item_id'].values)
        )
        return PyTorchDataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )

    def get_sampled_full_test_loader(self, batch_size, ratio=0.2):
        """
        빠른 검증을 위해 전체 유저 중 일부(ratio)만 샘플링하여 Full Evaluation을 수행하는 로더
        """
        unique_users = self.test_df['user_id'].unique()
        num_sample = int(len(unique_users) * ratio)
        if num_sample < 1:
            num_sample = 1
            
        # 에폭마다 동일한 유저 셋을 평가하기 위해 시드 고정
        rng = np.random.RandomState(42)
        sampled_users = rng.choice(unique_users, num_sample, replace=False)
        
        sampled_df = self.test_df[self.test_df['user_id'].isin(sampled_users)].copy()
        
        test_dataset = TensorDataset(
            torch.LongTensor(sampled_df['user_id'].values),
            torch.LongTensor(sampled_df['item_id'].values)
        )
        return PyTorchDataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )

    def get_full_valid_loader(self, batch_size):
        """Validation set에 대한 full ranking 로더"""
        valid_dataset = TensorDataset(
            torch.LongTensor(self.valid_df['user_id'].values),
            torch.LongTensor(self.valid_df['item_id'].values)
        )
        return PyTorchDataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )

    def get_sampled_valid_loader(self, batch_size, ratio=0.2):
        """
        빠른 검증을 위해 valid_df에서 유저 일부(ratio)만 샘플링하여 평가하는 로더.
        Test leakage 없음 (valid_df만 사용).
        """
        unique_users = self.valid_df['user_id'].unique()
        num_sample = max(1, int(len(unique_users) * ratio))
        
        # 에폭마다 동일한 유저 셋을 평가하기 위해 시드 고정
        rng = np.random.RandomState(42)
        sampled_users = rng.choice(unique_users, num_sample, replace=False)
        
        sampled_df = self.valid_df[self.valid_df['user_id'].isin(sampled_users)].copy()
        
        valid_dataset = TensorDataset(
            torch.LongTensor(sampled_df['user_id'].values),
            torch.LongTensor(sampled_df['item_id'].values)
        )
        return PyTorchDataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )

    def get_validation_loader(self, batch_size):
        """config 설정에 맞는 검증용 로더 반환 (valid_df 사용)"""
        method = self.config['evaluation'].get('validation_method', 'full')
        if method == 'full':
            return self.get_full_valid_loader(batch_size)
        elif method == 'sampled':
            ratio = self.config['evaluation'].get('validation_sample_ratio', 0.2)
            return self.get_sampled_valid_loader(batch_size, ratio=ratio)
        else:
            raise ValueError(f"Unsupported validation method: {method}. Use 'full' or 'sampled'.")

    def get_final_loader(self, batch_size):
        """config 설정에 맞는 최종 평가용 로더 반환 (test_df 사용)"""
        method = self.config['evaluation'].get('final_method', 'full')
        if method == 'full':
            return self.get_full_test_loader(batch_size)
        else:
            raise ValueError(f"Unsupported final method: {method}. Use 'full' for final evaluation.")