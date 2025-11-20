import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    BaseModel 추상 클래스
    모든 추천 모델은 이 클래스를 상속받아 구현해야 합니다.
    """
    def __init__(self, config, data_loader):
        """
        Args:
            config (dict): 설정 파일에서 읽어온 하이퍼파라미터 및 설정값
            data_loader (object): 데이터 로더 객체 (사용자/아이템 수 등의 정보 포함)
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.data_loader = data_loader

        # 디바이스 설정
        device_str = config['device']
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)


    @abstractmethod
    def calc_loss(self, batch_data):
        """
        배치 데이터로부터 손실(loss)을 계산합니다.

        요구사항에 따라 아래 두 가지를 튜플 형태로 반환해야 합니다.
        1. (메인 손실, *보조 손실들): 첫 번째 값은 반드시 메인 최적화 대상이 되는 손실이어야 합니다.
        2. (추적할 파라미터 딕셔너리): TensorBoard나 로그에 기록할 모델 내부 파라미터입니다.

        Args:
            batch_data: 트레이너로부터 전달받는 미니배치 데이터

        Returns:
            tuple: (losses, params_to_log)
                - losses (tuple): (main_loss, aux_loss1, aux_loss2, ...)
                - params_to_log (dict): {'param_name': value, ...}
        """
        pass

    @abstractmethod
    def forward(self, users):
        """
        주어진 사용자들에 대한 아이템 추천 점수를 반환합니다.
        평가(evaluation) 시에 사용됩니다.

        Args:
            users (torch.Tensor): 사용자 ID 텐서

        Returns:
            torch.Tensor: 모든 아이템에 대한 사용자별 추천 점수 (batch_size, num_items)
        """
        pass

    @abstractmethod
    def predict_for_pairs(self, user_ids, item_ids):
        """
        주어진 사용자-아이템 쌍에 대한 예측 평점을 반환합니다.
        평가(evaluation) 시에 사용됩니다.

        Args:
            user_ids (torch.Tensor): 사용자 ID 텐서 (N,)
            item_ids (torch.Tensor): 아이템 ID 텐서 (N,)

        Returns:
            torch.Tensor: 예측 평점 텐서 (N,)
        """
        pass

    @abstractmethod
    def get_final_item_embeddings(self):
        """
        모델이 최종적으로 사용하는 아이템 임베딩을 반환합니다.
        LightGCN이나 CSAR처럼 임베딩을 변환하는 모델은 이 메소드를 오버라이드해야 합니다.
        """
        pass

    def __str__(self):
        """
        모델의 이름과 주요 하이퍼파라미터를 문자열로 반환합니다.
        """
        return f"{self.__class__.__name__}({self.config['model']})"

    def sample_hard_negatives(self, users, pos_items, num_hard_negatives):
        """
        주어진 사용자-긍정 아이템 쌍에 대해 하드 네거티브 아이템을 샘플링합니다.
        """
        # 모델을 평가 모드로 전환 (임시)
        self.eval()
        
        with torch.no_grad():
            # 1. 모든 아이템에 대한 사용자별 예측 점수 계산
            all_scores = self.forward(users) # (batch_size, n_items)

            # 2. 이미 상호작용한 아이템 마스킹
            users_list = users.cpu().numpy()
            pos_items_list = pos_items.squeeze(1).cpu().numpy()

            for i, user_id in enumerate(users_list):
                # 사용자가 상호작용한 모든 아이템 (긍정 아이템 포함)
                interacted_items = list(self.data_loader.user_history[user_id])
                
                # 긍정 아이템 점수를 매우 낮은 값으로 설정하여 샘플링에서 제외
                all_scores[i, interacted_items] = -torch.inf # 이미 상호작용한 아이템 제외
                
                # 현재 긍정 아이템도 제외 (이미 interacted_items에 포함되어 있을 수 있지만 명시적으로)
                all_scores[i, pos_items_list[i]] = -torch.inf

            # 3. 마스킹된 점수 중에서 가장 높은 점수를 가진 아이템 선택
            # topk는 가장 큰 k개의 값을 반환하므로, -torch.inf로 마스킹된 아이템은 선택되지 않음
            _, hard_neg_indices = torch.topk(all_scores, k=num_hard_negatives, dim=1)
            
        # 모델을 다시 학습 모드로 전환
        self.train()
        
        return hard_neg_indices # (batch_size, num_hard_negatives)

