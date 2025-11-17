import torch
import torch.optim as optim
import yaml
import os
import numpy as np
from tqdm import tqdm
import collections
import json
import time
import copy

from .evaluation import evaluate_metrics
from .utils import plot_results

class Trainer:
    """
    추천 모델의 학습 과정을 관리하는 클래스.
    """
    def __init__(self, config, model, data_loader):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.device = model.device
        self.model.to(self.device)

        self.optimizer = self._get_optimizer(config['train']['optimizer'])
        
        model_name = self.config['model']['name']
        dataset_name = self.config['dataset_name']
        run_name = self.config.get('run_name')
        
        base_path = os.path.join('trained_model', dataset_name)
        
        if run_name and run_name != 'default':
            experiment_folder_name = f"{model_name}_{run_name}"
            self.output_path = os.path.join(base_path, experiment_folder_name)
        else:
            self.output_path = os.path.join(base_path, model_name)
            
        os.makedirs(self.output_path, exist_ok=True)

        with open(os.path.join(self.output_path, 'config.yaml'), 'w') as f:
            config_to_save = copy.deepcopy(self.config)
            config_to_save.pop('run_name', None)
            yaml.dump(config_to_save, f, default_flow_style=False)

        self.epochs = config['train']['epochs']
        self.early_stop_patience = config['train']['early_stop_patience']
        
        self.train_losses = collections.defaultdict(list)
        self.eval_metrics = collections.defaultdict(list)
        self.tracked_params = collections.defaultdict(list)

        self.best_metric_value = -float('inf')
        self.patience_counter = 0
        
        # [최적화] 데이터 로더를 미리 생성
        print("Initializing data loaders...")
        self.train_loader = self.data_loader.get_train_loader(self.config['train']['batch_size'])
        self.validation_loader = self.data_loader.get_validation_loader(self.config['train']['batch_size'] * 2)
        print("Data loaders initialized.")


    def _get_optimizer(self, optimizer_name):
        lr = self.config['train']['lr']
        weight_decay = self.config['train']['l2_regularization']
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(self):
        print(f"Training started on device: {self.device}")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = collections.defaultdict(float)
            epoch_params = collections.defaultdict(list)

            for batch_data in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

                if self.config['train'].get('loss_type') == 'pairwise' and self.config['train'].get('use_hard_negatives', False):
                    with torch.no_grad():
                        num_hard_negatives = self.config['train'].get('num_hard_negatives', 1)
                        hard_neg_items = self.model.sample_hard_negatives(
                            batch_data['user_id'], 
                            batch_data['pos_item_id'],
                            num_hard_negatives
                        )
                        batch_data['neg_item_id'] = hard_neg_items

                self.optimizer.zero_grad()
                losses_tuple, params_to_log = self.model.calc_loss(batch_data)

                total_loss_for_backward = sum(losses_tuple)
                total_loss_for_backward.backward()
                self.optimizer.step()

                epoch_losses['total_loss'] += total_loss_for_backward.item()
                for i, l in enumerate(losses_tuple):
                    epoch_losses[f'loss_{i}'] += l.item()
                
                if params_to_log:
                    for k, v in params_to_log.items():
                        epoch_params[k].append(v)

            avg_total_loss = epoch_losses['total_loss'] / len(self.train_loader)
            self.train_losses['total_loss'].append(avg_total_loss)
            avg_main_loss = epoch_losses['loss_0'] / len(self.train_loader) if 'loss_0' in epoch_losses else 0.0
            self.train_losses['loss_0'].append(avg_main_loss)
            for k, v in epoch_losses.items():
                if k != 'total_loss' and k != 'loss_0':
                    self.train_losses[k].append(v / len(self.train_loader))

            if epoch_params:
                for k, v_list in epoch_params.items():
                    self.tracked_params[k].append(np.mean(v_list))
            
            main_metric = self.config['evaluation']['main_metric']
            main_metric_k = self.config['evaluation']['main_metric_k']
            
            # [최적화] 미리 생성된 검증 로더를 전달
            current_metrics = self.evaluate(
                loader=self.validation_loader,
                is_final_evaluation=False,
                override_metrics=[main_metric],
                override_top_k=[main_metric_k]
            )
            
            for k, v in current_metrics.items():
                self.eval_metrics[k].append(v)
            
            main_metric_name = f"{main_metric}@{main_metric_k}"
            main_metric_value = current_metrics.get(main_metric_name, 0.0)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_total_loss:.4f} (Main: {avg_main_loss:.4f}) - Val {main_metric_name}: {main_metric_value:.4f}")

            if self._check_early_stopping(current_metrics):
                break
        
        print("Training finished. Performing final evaluation...")
        # 최종 평가는 loader=None으로 호출하여 내부에서 로더를 생성하도록 함
        self.evaluate(is_final_evaluation=True)
        
        self._visualize_results()

    def evaluate(self, loader=None, is_final_evaluation=False, override_metrics=None, override_top_k=None):
        self.model.eval()
        
        eval_config = copy.deepcopy(self.config)
        
        # loader가 제공되지 않은 경우에만 새로 생성 (주로 최종 평가 시)
        if loader is None:
            print("Creating a new loader for evaluation...")
            if is_final_evaluation:
                loader = self.data_loader.get_final_loader(self.config['train']['batch_size'] * 2)
            else: # 이론적으로는 이 분기에 올 수 없지만, 안전을 위해 추가
                loader = self.data_loader.get_validation_loader(self.config['train']['batch_size'] * 2)


        if is_final_evaluation:
            eval_config['evaluation']['method'] = self.config['evaluation'].get('final_method', 'full')
            eval_config['evaluation']['metrics'] = self.config['evaluation'].get('metrics', ['HitRate', 'NDCG'])
            eval_config['evaluation']['top_k'] = self.config['evaluation'].get('top_k', [10, 20])
        else:
            eval_config['evaluation']['method'] = self.config['evaluation'].get('validation_method', 'uni99')
            eval_config['evaluation']['metrics'] = override_metrics if override_metrics else self.config['evaluation'].get('metrics', ['NDCG'])
            eval_config['evaluation']['top_k'] = override_top_k if override_top_k else self.config['evaluation'].get('top_k', [10])

        # [최적화] loader가 주어지면 test_loader 인자로 전달
        current_metrics = evaluate_metrics(self.model, self.data_loader, eval_config, self.device, test_loader=loader)
        
        if is_final_evaluation:
            print(f"Final Evaluation: {current_metrics}")
            metrics_path = os.path.join(self.output_path, 'final_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(current_metrics, f, indent=4)
            print(f"Final metrics saved to {metrics_path}")

        return current_metrics

    def _check_early_stopping(self, current_metrics):
        main_metric_name = f"{self.config['evaluation']['main_metric']}@{self.config['evaluation']['main_metric_k']}"
        current_main_metric = current_metrics.get(main_metric_name)

        if current_main_metric is None:
            print(f"[Warning] Main metric '{main_metric_name}' not found in evaluation results. Skipping early stopping.")
            return False

        if current_main_metric > self.best_metric_value:
            self.best_metric_value = current_main_metric
            self.patience_counter = 0
            self._save_checkpoint()
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stop_patience:
            print(f"Early stopping triggered at epoch {len(self.train_losses['total_loss'])}")
            return True
        return False

    def _save_checkpoint(self):
        checkpoint_file = os.path.join(self.output_path, "best_model.pt")
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Best performance updated. Checkpoint saved to {checkpoint_file}")

    def _visualize_results(self):
        loss_plot_dict = {k: v for k, v in self.train_losses.items() if k.startswith('loss_')}
        secondary_keys = [k for k in loss_plot_dict.keys() if k != 'loss_0']
        plot_results(
            data_dict=loss_plot_dict,
            title="Training Losses Over Epochs",
            xlabel="Epoch",
            ylabel="Main Loss (loss_0)",
            file_path=os.path.join(self.output_path, "loss_plot.png"),
            secondary_y_keys=secondary_keys
        )
        
        if self.tracked_params:
            plot_results(
                data_dict=self.tracked_params,
                title="Tracked Parameters Over Epochs",
                xlabel="Epoch",
                ylabel="Parameter Value",
                file_path=os.path.join(self.output_path, "params_plot.png")
            )
        
        if self.eval_metrics:
            plot_results(
                data_dict=self.eval_metrics,
                title="Validation Metrics Over Epochs",
                xlabel="Epoch",
                ylabel="Metric Value",
                file_path=os.path.join(self.output_path, "metrics_plot.png")
            )

