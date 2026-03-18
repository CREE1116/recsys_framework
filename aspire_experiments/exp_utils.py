import os
import sys
import yaml
import torch
import numpy as np
import optuna
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import (
    SVDCacheManager, 
    EVDCacheManager, 
    GramEigenCacheManager, 
    GramMatrixCacheManager, 
    CholeskyCacheManager
)

def load_config(dataset_name):
    """YAML 파일을 로드하여 설정을 반환합니다."""
    # dataset_name can be a path or a name in configs/dataset/
    if not dataset_name.endswith('.yaml'):
        config_path = f"configs/dataset/{dataset_name}.yaml"
    else:
        config_path = dataset_name
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # model name and evaluation config are required by DataLoader/Evaluation
    if 'model' not in config:
        config['model'] = {'name': 'aspire'}
    if 'evaluation' not in config:
        config['evaluation'] = {
            'method': 'full',
            'top_k': [10],
            'metrics': ['NDCG'],
            'validation_method': 'full'
        }
    elif 'validation_method' not in config['evaluation']:
        config['evaluation']['validation_method'] = 'full'
    
    return config

def get_loader_and_svd(dataset_name, k=None, target_energy=None, seed=42):
    """DataLoader와 SVD 데이터를 초기화합니다. target_energy는 EVD 경로에서 무시됩니다."""
    # Set global seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Memory Management: 새로운 데이터셋 로드 전 기존 캐시 비움
    EVDCacheManager.clear()
    SVDCacheManager.clear()
    # Actually, Cholesky and Eigen caches are the hungry ones.
    GramEigenCacheManager.clear()
    GramMatrixCacheManager.clear()
    CholeskyCacheManager.clear()
    config = load_config(dataset_name)
    if seed is not None:
        config['seed'] = seed
    loader = DataLoader(config)
    
    # R (Interaction Matrix) 생성
    rows = loader.train_df['user_id'].values
    cols = loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    R = csr_matrix((vals, (rows, cols)), shape=(loader.n_users, loader.n_items))
    
    # EVD 계산 (ASPIRE용 고윳값 분해 경로)
    manager = EVDCacheManager()
    U, S, V, _ = manager.get_evd(R, k=k, dataset_name=config["dataset_name"])
    
    return loader, R, S, V, config

def get_trimmed_data(x, y, trim_range=(0.05, 0.05)):
    """양끝 trim_range 만큼 데이터를 제거합니다."""
    n = len(x)
    low, high = trim_range
    start_idx = int(n * low)
    end_idx = n - int(n * high)
    if end_idx - start_idx >= 2:
        return x[start_idx:end_idx], y[start_idx:end_idx]
    return x, y

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_eval_config(loader, override: dict = None) -> dict:
    """
    configs/evaluation.yaml 기준으로 eval_config를 구성한다.
    loader.config에 dataset 고유 설정이 있으면 우선 적용하고,
    override dict로 추가 덮어씌우기 가능.

    Returns a dict compatible with evaluate_metrics().
    """
    eval_yaml_path = "configs/evaluation.yaml"
    eval_cfg = {}
    if os.path.exists(eval_yaml_path):
        with open(eval_yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        eval_cfg = raw.get("evaluation", {})

    # loader.config의 evaluation 섹션이 있으면 덮어씌움
    loader_eval = loader.config.get("evaluation", {})
    eval_cfg.update(loader_eval)

    # 전체 config 병합 (evaluate_metrics가 loader.config 기반으로 동작)
    config = loader.config.copy()
    config["metrics"]           = eval_cfg.get("metrics",
                                                ["NDCG", "Recall", "Coverage"])
    config["top_k"]             = eval_cfg.get("top_k", [10])
    config["long_tail_percent"] = eval_cfg.get("long_tail_percent", 0.8)

    if override:
        config.update(override)

    return config


class AspireHPO:
    """
    프레임워크의 BayesianOptimizer(scripts/bayesian_opt.py) 패턴을 따르는 경량 HPO 클래스.
    커스텀 objective function과 함께 사용하며 아래 기능을 제공:
      - TPESampler (프레임워크와 동일한 sampler)
      - EarlyStoppingCallback (patience 기반)
      - save_results() : CSV + 최적화 히스토리/중요도/slice 플롯

    params_spec 형식 (bayesian_opt.py와 동일):
        [
            {'name': 'alpha', 'type': 'float', 'range': '0.1 1e6', 'log': True},
            {'name': 'beta',  'type': 'float', 'range': '0.0 1.5'},
        ]
    """

    def __init__(self, params_spec, n_trials=30, patience=10, seed=42, direction="maximize"):
        self.params_spec = params_spec
        self.n_trials = n_trials
        self.patience = patience
        self.seed = seed
        self.maximize = (direction == "maximize")
        self.study = None

    def _suggest(self, trial, p_def):
        name   = p_def['name']
        p_type = p_def.get('type', 'float')
        p_range = p_def.get('range') or p_def.get('choices')
        p_log  = p_def.get('log', False)

        if p_range is None:
            raise KeyError(f"Parameter '{name}' must have 'range' or 'choices'.")

        if p_type == 'float':
            low, high = map(float, p_range.split())
            return trial.suggest_float(name, low, high, log=p_log)
        elif p_type == 'int':
            low, high = map(int, p_range.split())
            return trial.suggest_int(name, low, high, log=p_log)
        elif p_type == 'categorical':
            choices = p_range if isinstance(p_range, list) else p_range.split()
            return trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown parameter type: {p_type}")

    def search(self, objective_fn, study_name="HPO", output_dir=None):
        """
        objective_fn: callable(params_dict) -> float
            params_dict는 {'alpha': ..., 'beta': ...} 형태의 dict.
        """
        def _wrapped_objective(trial):
            params = {p['name']: self._suggest(trial, p) for p in self.params_spec}
            return objective_fn(params)

        maximize = self.maximize
        patience = self.patience

        class EarlyStoppingCallback:
            """bayesian_opt.py와 동일한 patience 기반 조기 종료."""
            def __init__(self):
                self.best_score = -float('inf') if maximize else float('inf')
                self.no_improve_count = 0

            def __call__(self, study, trial):
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    return
                current = trial.value
                is_better = current > self.best_score if maximize else current < self.best_score
                if is_better:
                    self.best_score = current
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                if patience and self.no_improve_count >= patience:
                    print(f"[Early Stopping] {patience}회 연속 미향상. 탐색 종료.")
                    study.stop()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction="maximize" if maximize else "minimize",
            sampler=sampler,
            study_name=study_name,
        )

        print(f"[HPO:{study_name}] n_trials={self.n_trials}, patience={patience}, seed={self.seed}")
        self.study.optimize(
            _wrapped_objective,
            n_trials=self.n_trials,
            callbacks=[EarlyStoppingCallback()],
            show_progress_bar=True,
        )

        best = self.study.best_params
        best_val = self.study.best_value
        print(f"[HPO:{study_name}] Best params={best}  Val NDCG={best_val:.4f}")

        if output_dir:
            self.save_results(output_dir)

        return best, best_val

    def save_results(self, output_dir):
        """bayesian_opt.py의 save_results()와 동일한 패턴으로 결과 저장."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        # CSV
        try:
            df = self.study.trials_dataframe()
            df.to_csv(os.path.join(output_dir, 'hpo_trials.csv'), index=False)
        except Exception as e:
            print(f"[HPO] CSV 저장 실패: {e}")

        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return

        values = [t.value for t in trials]
        agg = np.maximum.accumulate if self.maximize else np.minimum.accumulate
        best_values = list(agg(values))

        # Optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker='o', alpha=0.5, label='Objective')
        plt.plot(best_values, color='red', linewidth=2, label='Best')
        plt.xlabel('Trial'); plt.ylabel('NDCG@10'); plt.title('Optimization History')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=120)
        plt.close()

        # Parameter importance (파라미터 2개 이상일 때만)
        if len(self.params_spec) > 1:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                plt.figure(figsize=(8, 5))
                plt.barh(list(importance.keys()), list(importance.values()))
                plt.xlabel('Importance'); plt.title('Hyperparameter Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'param_importance.png'), dpi=120)
                plt.close()
            except Exception as e:
                print(f"[HPO] 중요도 플롯 실패: {e}")

        # Slice plots
        for p_def in self.params_spec:
            p_name = p_def['name']
            try:
                x = [t.params[p_name] for t in trials if p_name in t.params]
                y = [t.value        for t in trials if p_name in t.params]
                plt.figure(figsize=(8, 5))
                plt.scatter(x, y, alpha=0.6)
                plt.xlabel(p_name); plt.ylabel('NDCG@10'); plt.title(f'Slice: {p_name}')
                if p_def.get('log'):
                    plt.xscale('log')
                plt.grid(True, alpha=0.3)
                safe_name = p_name.replace('.', '_')
                plt.savefig(os.path.join(output_dir, f'slice_{safe_name}.png'), dpi=120)
                plt.close()
            except Exception as e:
                print(f"[HPO] Slice plot 실패 ({p_name}): {e}")

        print(f"[HPO] 결과 저장 완료: {output_dir}")
