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
    def __init__(self, df, n_items, user_history, loss_type, num_negatives, sampling_weights=None, train_user_history=None):
        self.df = df
        self.n_items = n_items
        self.user_history = user_history
        # [RecBole 표준] 학습 시에는 train_user_history만 사용 (데이터 누수 방지)
        self.train_user_history = train_user_history if train_user_history is not None else user_history
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
            # Negative Sampling 없음 - 그대로 반환
            return torch.utils.data.default_collate(batch)
        
        # Pairwise: Negative Sampling 수행 후 item_id -> pos_item_id로 변경
        users = [item['user_id'] for item in batch]
        pos_items = [item['item_id'] for item in batch]
        
        batch_size = len(batch)
        users_tensor = torch.tensor(users, dtype=torch.long)
        pos_items_tensor = torch.tensor(pos_items, dtype=torch.long)
        
        # [최적화] 벡터화된 Negative Sampling
        # 각 유저별로 필요한 수의 3배를 한번에 생성하여 효율 향상
        oversample_factor = 3
        total_candidates = batch_size * self.num_negatives * oversample_factor
        
        if self.sampling_weights is not None:
            # Popularity 기반 샘플링
            candidates = torch.multinomial(
                self.sampling_weights, 
                total_candidates, 
                replacement=True
            )
        else:
            # Uniform 샘플링
            candidates = torch.randint(0, self.n_items, (total_candidates,))
        
        candidates = candidates.view(batch_size, self.num_negatives * oversample_factor)
        
        # 각 유저별로 seen 아이템 필터링
        # [RecBole 표준] 학습 시에는 train_user_history만 사용 (데이터 누수 방지)
        neg_items_list = []
        for i in range(batch_size):
            u = users[i]
            user_seen = self.train_user_history.get(u, set())
            user_candidates = candidates[i].tolist()
            
            # seen 아이템 필터링
            valid_negs = [c for c in user_candidates if c not in user_seen]
            
            # 충분하지 않으면 추가 샘플링 (드문 경우)
            while len(valid_negs) < self.num_negatives:
                if self.sampling_weights is not None:
                    c = torch.multinomial(self.sampling_weights, 1, replacement=True).item()
                else:
                    c = torch.randint(0, self.n_items, (1,)).item()
                if c not in user_seen:
                    valid_negs.append(c)
            
            neg_items_list.append(valid_negs[:self.num_negatives])
        
        neg_items_tensor = torch.tensor(neg_items_list, dtype=torch.long)
        
        # [Fix] Broadcasting 버그 방지: user_id와 pos_item_id도 2D [B, 1]로 변환
        users_tensor = users_tensor.unsqueeze(1)
        pos_items_tensor = pos_items_tensor.unsqueeze(1)
        
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
        self.has_header = config.get('has_header', False)  # 헤더 유무
        
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
            self.valid_uni99_negatives = None

        self.sampling_weights = None
        if self.neg_sampling_strategy == 'popularity':
            print(f"Calculating popularity-based sampling weights (alpha={self.neg_sampling_alpha})...")
            # 캐시 로드 시 item_popularity가 없을 경우를 대비해 다시 계산 (일반적으로는 _process_data에서 처리됨)
            if not hasattr(self, 'item_popularity'):
                 self.item_popularity = self.train_df['item_id'].value_counts().sort_index().reindex(range(self.n_items), fill_value=0)

            counts = torch.FloatTensor(self.item_popularity.values)
            self.sampling_weights = torch.pow(counts, self.neg_sampling_alpha)
            self.sampling_weights /= self.sampling_weights.sum()
            
        if not hasattr(self, 'train_user_history'):
             print("Generating train_user_history from train_df...")
             self.train_user_history = self.train_df.groupby('user_id')['item_id'].apply(set).to_dict()

        if not hasattr(self, 'eval_user_history'):
             print("Generating eval_user_history (train+valid) for test evaluation...")
             eval_df = pd.concat([self.train_df, self.valid_df], ignore_index=True)
             self.eval_user_history = eval_df.groupby('user_id')['item_id'].apply(set).to_dict()

        # 검증 방식이 uni99인 경우 네거티브 샘플링 수행
        if self.config['evaluation'].get('validation_method') == 'uni99' or self.config['evaluation'].get('final_method') == 'uni99':
            # 둘 다 있어야 함을 확인
            if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None and \
               hasattr(self, 'valid_uni99_negatives') and self.valid_uni99_negatives is not None:
                print("Loaded pre-sampled uni99 negatives (valid & test) from cache.")
            else:
                self._pre_sample_test_negatives()
                self._save_to_cache()

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
        
        # 분할 방식 선택
        split_method = self.config.get('split_method', 'loo')
        if split_method == 'loo':
            self.train_df, self.valid_df, self.test_df = self._split_leave_one_out(self.df)
        elif split_method == 'temporal_ratio':
            train_ratio = self.config.get('train_ratio', 0.8)
            valid_ratio = self.config.get('valid_ratio', 0.1)
            self.train_df, self.valid_df, self.test_df = self._split_temporal_ratio(
                self.df, train_ratio=train_ratio, valid_ratio=valid_ratio
            )
        elif split_method == 'random':
            train_ratio = self.config.get('train_ratio', 0.8)
            valid_ratio = self.config.get('valid_ratio', 0.1)
            self.train_df, self.valid_df, self.test_df = self._split_random(
                self.df, train_ratio=train_ratio, valid_ratio=valid_ratio
            )
        else:
            raise ValueError(f"Unknown split_method: {split_method}. Use 'loo', 'temporal_ratio', or 'random'.")
        
        self.user_history = self.df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.train_user_history = self.train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        
        # [RecBole Alignment] eval_user_history for test masking (train + valid)
        eval_df = pd.concat([self.train_df, self.valid_df], ignore_index=True)
        self.eval_user_history = eval_df.groupby('user_id')['item_id'].apply(set).to_dict()

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
            'eval_user_history': self.eval_user_history,
            'item_popularity': self.item_popularity,
        }
        if hasattr(self, 'test_uni99_negatives') and self.test_uni99_negatives is not None:
            data_to_cache['test_uni99_negatives'] = self.test_uni99_negatives
        if hasattr(self, 'valid_uni99_negatives') and self.valid_uni99_negatives is not None:
            data_to_cache['valid_uni99_negatives'] = self.valid_uni99_negatives
            
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data_to_cache, f)

    def _load_from_cache(self):
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            for key, value in cached_data.items():
                setattr(self, key, value)

    def _load_data(self):
        # engine='python'을 사용하여 다양한 구분자 지원 강화
        if self.has_header:
            # 헤더가 있는 경우: 첫 줄을 헤더로 인식하고 columns로 이름 변경
            df = pd.read_csv(self.data_path, sep=self.separator, header=0, engine='python')
            df.columns = self.columns[:len(df.columns)]  # config columns로 이름 변경
            return df
        else:
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
        # [BUG FIX] sorted() 필수 - unique()는 비결정적 순서를 반환하므로 정렬하여 일관성 보장
        unique_users = sorted(self.df['user_id'].unique())
        unique_items = sorted(self.df['item_id'].unique())
        
        self.user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        self.item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        
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
        # Timestamp 컬럼 확인 (없으면 인덱스 기반으로 정렬)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id'])
        else:
            # timestamp 없으면 현재 순서 유지 (random split과 유사)
            df_sorted = df.sort_values(by=['user_id', 'item_id'])
        
        # Test: 마지막 아이템
        test_df = df_sorted.groupby('user_id').tail(1)
        
        # user_id + item_id 만으로 merge (rating 컬럼 의존성 제거)
        merge_keys = ['user_id', 'item_id']
        
        # Valid: 끝에서 2번째 아이템 (Test 제외 후 마지막)
        remaining_after_test = df_sorted.merge(
            test_df[merge_keys], on=merge_keys, 
            how='left', indicator=True
        )
        remaining_after_test = remaining_after_test[remaining_after_test['_merge'] == 'left_only'].drop(columns=['_merge'])
        valid_df = remaining_after_test.groupby('user_id').tail(1)
        
        # Train: Test, Valid 모두 제외
        eval_df = pd.concat([test_df, valid_df], ignore_index=True)
        train_df = df.merge(
            eval_df[merge_keys], on=merge_keys, 
            how='left', indicator=True
        )
        train_df = train_df[train_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        return train_df, valid_df, test_df

    def _split_temporal_ratio(self, df, train_ratio=0.8, valid_ratio=0.1):
        """
        User-level Temporal Ratio Split (유저별 시간순 비율 분할)
        - 유저별로 시간순 정렬 후 비율에 따라 분할
        - Train: 앞에서 train_ratio% (과거)
        - Valid: 다음 valid_ratio%
        - Test: 나머지 (1 - train_ratio - valid_ratio)% (미래)
        
        Note: 학술 연구 표준 방식 (RecBole, LightGCN 등)
        
        Args:
            df: 전체 데이터프레임
            train_ratio: 학습 데이터 비율 (default: 0.8)
            valid_ratio: 검증 데이터 비율 (default: 0.1)
        """
        # Timestamp 기준 정렬
        df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id'])
        
        train_list = []
        valid_list = []
        test_list = []
        
        # 유저별로 분할
        for user_id, group in df_sorted.groupby('user_id'):
            n = len(group)
            
            # 최소 3개 상호작용 필요 (train, valid, test 각 1개)
            if n < 3:
                train_list.append(group)
                continue
            
            train_end = max(1, int(n * train_ratio))
            valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
            
            # 최소 1개씩 보장
            if valid_end >= n:
                valid_end = n - 1
            if train_end >= valid_end:
                train_end = valid_end - 1
            
            train_list.append(group.iloc[:train_end])
            valid_list.append(group.iloc[train_end:valid_end])
            test_list.append(group.iloc[valid_end:])
        
        train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
        valid_df = pd.concat(valid_list, ignore_index=True) if valid_list else pd.DataFrame()
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        
        return train_df, valid_df, test_df

    def _split_random(self, df, train_ratio=0.8, valid_ratio=0.1):
        """
        Random Ratio Split (랜덤 비율 분할) - timestamp 없는 데이터셋용
        - 유저별로 상호작용을 셔플 후 비율에 따라 분할
        - Train: 앞에서 train_ratio%
        - Valid: 다음 valid_ratio%
        - Test: 나머지 (1 - train_ratio - valid_ratio)%
        
        Args:
            df: 전체 데이터프레임
            train_ratio: 학습 데이터 비율 (default: 0.8)
            valid_ratio: 검증 데이터 비율 (default: 0.1)
        """
        train_list = []
        valid_list = []
        test_list = []
        
        # 유저별로 분할
        for user_id, group in df.groupby('user_id'):
            # 랜덤 셔플
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(group)
            
            # 최소 3개 상호작용 필요 (train, valid, test 각 1개)
            if n < 3:
                # 상호작용이 부족하면 전부 train으로
                train_list.append(group)
                continue
            
            train_end = max(1, int(n * train_ratio))
            valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
            
            # 최소 1개씩 보장
            if valid_end >= n:
                valid_end = n - 1
            if train_end >= valid_end:
                train_end = valid_end - 1
            
            train_list.append(group.iloc[:train_end])
            valid_list.append(group.iloc[train_end:valid_end])
            test_list.append(group.iloc[valid_end:])
        
        train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
        valid_df = pd.concat(valid_list, ignore_index=True) if valid_list else pd.DataFrame()
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        
        return train_df, valid_df, test_df

    def _pre_sample_test_negatives(self):
        print("Pre-sampling negative items for uni99 evaluation...")
        
        # Test Negatives
        self.test_uni99_negatives = {}
        test_user_item_pairs = self.test_df[['user_id', 'item_id']].to_dict('records')
        for row in tqdm(test_user_item_pairs, desc="Sampling test uni99 negatives"):
            user_id, pos_item_id = row['user_id'], row['item_id']
            seen_items = set(self.eval_user_history.get(user_id, set())) # Test evaluation uses eval_user_history (train+valid)
            seen_items.add(pos_item_id)
            
            negative_items = []
            while len(negative_items) < 99:
                candidates = np.random.randint(0, self.n_items, size=200)
                valid_negatives = list(set(candidates) - seen_items)
                negative_items.extend(valid_negatives)
                seen_items.update(valid_negatives)
            self.test_uni99_negatives[user_id] = negative_items[:99]

        # Validation Negatives
        self.valid_uni99_negatives = {}
        valid_user_item_pairs = self.valid_df[['user_id', 'item_id']].to_dict('records')
        for row in tqdm(valid_user_item_pairs, desc="Sampling valid uni99 negatives"):
            user_id, pos_item_id = row['user_id'], row['item_id']
            seen_items = set(self.train_user_history.get(user_id, set())) # Valid evaluation uses train_user_history
            seen_items.add(pos_item_id)
            
            negative_items = []
            while len(negative_items) < 99:
                candidates = np.random.randint(0, self.n_items, size=200)
                valid_negatives = list(set(candidates) - seen_items)
                negative_items.extend(valid_negatives)
                seen_items.update(valid_negatives)
            self.valid_uni99_negatives[user_id] = negative_items[:99]

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
            self.loss_type, self.num_negatives, self.sampling_weights,
            train_user_history=self.train_user_history
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
        elif method == 'uni99':
            return self.get_uni99_valid_loader(batch_size)
        else:
            raise ValueError(f"Unsupported validation method: {method}. Use 'full', 'sampled', or 'uni99'.")

    def get_uni99_valid_loader(self, batch_size):
        """uni99 방식의 검증 로더"""
        dataset = Uni99RecSysDataset(self.valid_df, self.valid_uni99_negatives)
        return PyTorchDataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0
        )

    def get_uni99_test_loader(self, batch_size):
        """uni99 방식의 테스트 로더"""
        dataset = Uni99RecSysDataset(self.test_df, self.test_uni99_negatives)
        return PyTorchDataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            persistent_workers=self.config.get('num_workers', 4) > 0
        )

    def get_final_loader(self, batch_size):
        """config 설정에 맞는 최종 평가용 로더 반환 (test_df 사용)"""
        method = self.config['evaluation'].get('final_method', 'full')
        if method == 'full':
            return self.get_full_test_loader(batch_size)
        elif method == 'uni99':
            return self.get_uni99_test_loader(batch_size)
        else:
            raise ValueError(f"Unsupported final method: {method}. Use 'full' or 'uni99'.")