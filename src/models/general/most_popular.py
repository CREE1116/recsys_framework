import torch
import torch.nn as nn
from ..base_model import BaseModel

class MostPopular(BaseModel):
    """
    가장 인기 있는 아이템을 추천하는 간단한 베이스라인 모델.
    이 모델은 학습되지 않습니다.
    """
    def __init__(self, config, data_loader):
        super(MostPopular, self).__init__(config, data_loader)
        
        # main.py에서 fit()을 명시적으로 호출하므로, __init__에서는 데이터 로드만 준비
        self.popularity_scores = None
        
        print("MostPopular model initialized.")

    def fit(self, data_loader):
        """
        MostPopular 모델의 아이템 인기도 점수를 계산합니다.
        이 메소드는 main.py에서 학습이 필요 없는 모델에 대해 호출됩니다.
        """
        # data_loader는 self.data_loader와 동일한 인스턴스
        item_popularity = data_loader.item_popularity
        self.popularity_scores = torch.FloatTensor(item_popularity.values).to(self.device)
        print("MostPopular model fitted successfully.")

    def forward(self, users):
        """
        모든 사용자에 대해 동일한 인기도 점수를 반환합니다.
        """
        # BUG FIX: fit()이 호출된 후에만 popularity_scores에 접근하도록 변경
        if self.popularity_scores is None:
            raise RuntimeError("Model has not been fitted. Call model.fit(data_loader) first.")
            
        batch_size = users.size(0)
        return self.popularity_scores.unsqueeze(0).repeat(batch_size, 1)

    def predict_for_pairs(self, user_ids, item_ids):
        """
        주어진 아이템들의 인기도 점수를 반환합니다.
        """
        if self.popularity_scores is None:
            raise RuntimeError("Model has not been fitted. Call model.fit(data_loader) first.")
        
        return self.popularity_scores[item_ids]

    def get_embeddings(self):
        """
        MostPopular 모델은 임베딩이 없으므로, None을 반환합니다.
        evaluation.py에서 이를 자동으로 건너뜁니다.
        """
        return None, None

    def get_final_item_embeddings(self):
        """
        MostPopular 모델은 임베딩이 없으므로, 인기도 점수를 pseudo-embedding으로 반환합니다.
        그러나 get_embeddings()가 None을 반환하므로 사실상 사용되지 않습니다.
        """
        return self.popularity_scores.unsqueeze(1).detach()

    def calc_loss(self, batch_data):
        """
        이 모델은 학습되지 않으므로 손실은 0입니다.
        """
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return "MostPopular"
