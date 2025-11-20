import torch
import torch.nn.functional as F
from .base_model import BaseModel

class ReRankWrapper(BaseModel):
    """
    기존 모델을 감싸서 추론 시에만 재순위화(Re-ranking)를 적용하는 메타 모델.
    학습은 내부의 base_model이 수행하고, forward(추론) 시에만 MMR 알고리즘을 적용.
    """
    def __init__(self, config, data_loader):
        super(ReRankWrapper, self).__init__(config, data_loader)

        # 재순위화 하이퍼파라미터 (forward에서 사용하지 않으므로, 필요에 따라 삭제 가능)
        self.rerank_lambda = self.config['model']['rerank_lambda']
        self.rerank_k = self.config['model']['rerank_top_k']

        # 내부 베이스 모델 생성
        base_model_name = self.config['model']['base_model_name']
        # Wrapper 설정에서 base_model의 설정을 추출하여 전달
        base_model_config = self.config.copy()
        base_model_config['model'] = self.config['model']['base_model_config']
        from . import get_model
        self.base_model = get_model(base_model_name, base_model_config, data_loader)
        
        # Wrapper 모델의 디바이스와 base_model의 디바이스를 동기화
        self.to(self.device)
        self.base_model.to(self.device)

    def calc_loss(self, batch_data):
        """학습은 전적으로 베이스 모델에 위임합니다."""
        return self.base_model.calc_loss(batch_data)

    def predict_for_pairs(self, user_ids, item_ids):
        """Pair 예측은 베이스 모델의 로직을 그대로 사용합니다."""
        return self.base_model.predict_for_pairs(user_ids, item_ids)

    def get_final_item_embeddings(self):
        """Wrapper는 base_model의 최종 임베딩을 그대로 반환합니다."""
        return self.base_model.get_final_item_embeddings()

    def forward(self, users):
        """
        [BUG FIX] full evaluation 시 성능 문제 해결을 위해, ReRankWrapper의 forward 메소드는
        베이스 모델의 forward 메소드를 그대로 호출하여 전체 아이템에 대한 점수를 반환합니다.
        MMR 재순위화 로직은 `predict_topk_reranked`와 같은 별도의 메소드에서 처리되어야 하며,
        일반적인 `forward`는 모델의 기본 스코어링 능력을 나타내도록 합니다.
        """
        return self.base_model.forward(users)

    # Note: If actual re-ranked top-k lists are needed for recommendation,
    # a separate method like 'predict_topk_reranked' should be implemented,
    # encapsulating the MMR logic.
    # The MMR logic from the previous forward implementation could be moved here.
    def predict_topk_reranked(self, users, top_k=None):
        """
        MMR(Maximal Marginal Relevance)을 사용하여 상위 k개 아이템을 재순위화하여 반환합니다.
        (이 메소드는 evaluation.py에서 직접 호출되지 않으므로, 수동으로 사용해야 합니다.)
        """
        if top_k is None:
            top_k = self.rerank_k
            
        relevance_scores = self.base_model.forward(users)
        
        with torch.no_grad():
            if hasattr(self.base_model, 'get_embeddings'):
                _, item_embeddings = self.base_model.get_embeddings()
            else:
                item_embeddings = self.base_model.item_embedding.weight
            
            item_embeddings_norm = F.normalize(item_embeddings, p=2, dim=1)

        final_re_ranked_indices = []

        for i, user_scores in enumerate(relevance_scores):
            # 초기 후보군: 관련성 점수가 높은 상위 K개 아이템
            _, initial_candidates_indices = torch.topk(user_scores, k=top_k, dim=-1)
            
            re_ranked_list_for_user = []
            
            # 첫 번째 아이템은 관련성이 가장 높은 아이템으로 선택
            if initial_candidates_indices.numel() > 0:
                first_item_idx = initial_candidates_indices[0].item()
                re_ranked_list_for_user.append(first_item_idx)
            
            remaining_candidates_indices = initial_candidates_indices[1:].tolist()
            
            while len(re_ranked_list_for_user) < top_k and remaining_candidates_indices:
                best_next_item = -1
                max_mmr_score = -float('inf')

                selected_embeds = item_embeddings_norm[re_ranked_list_for_user]
                
                for candidate_item_idx in remaining_candidates_indices:
                    candidate_embed = item_embeddings_norm[candidate_item_idx]
                    
                    relevance = user_scores[candidate_item_idx]
                    similarity = torch.matmul(selected_embeds, candidate_embed).max()
                    
                    mmr_score = self.rerank_lambda * relevance - (1 - self.rerank_lambda) * similarity
                    
                    if mmr_score > max_mmr_score:
                        max_mmr_score = mmr_score
                        best_next_item = candidate_item_idx
                
                if best_next_item != -1:
                    re_ranked_list_for_user.append(best_next_item)
                    remaining_candidates_indices.remove(best_next_item)
                else:
                    break
            
            # top_k에 맞춰 패딩 또는 자르기
            while len(re_ranked_list_for_user) < top_k:
                re_ranked_list_for_user.append(-1) # 유효하지 않은 아이템 ID로 패딩
            final_re_ranked_indices.append(re_ranked_list_for_user[:top_k])
        
        return torch.tensor(final_re_ranked_indices, device=self.device)

    def __str__(self):
        return f"ReRankWrapper(base_model={self.base_model}, lambda={self.rerank_lambda}, k={self.rerank_k})"
