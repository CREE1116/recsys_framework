# 모델 이름과 클래스를 매핑하는 레지스트리
# main.py에서 config 파일의 모델 이름을 기반으로 동적으로 클래스를 가져오는 데 사용됩니다.

# General Models
from .general.mf import MF
from .general.lightgcn import LightGCN
from .general.most_popular import MostPopular
from .general.item_knn import ItemKNN
from .general.neumf import NeuMF
from .general.rerank_wrapper import ReRankWrapper
from .general.ACF_NLL import ACF_NLL, ACF_BPR
from .general.softplusmf import softplusMF
from .general.ease import EASE
from .general.protomf import ProtoMF
from .general.multivae import MultiVAE

# CSAR Models
from .csar.CSAR import CSAR
from .csar.CSAR_Sampled import CSAR_Sampled
from .csar.CSAR_R import CSAR_R
from .csar.CSAR_BPR import CSAR_BPR
from .csar.CSAR_R_BPR import CSAR_R_BPR

# Legacy Models (Commented out)
# from .legacy.CSAR_R_Softmax import CSAR_R_Softmax
# from .legacy.CSAR_contrastive import CSAR_contrastive
# from .legacy.CSAR_Deep import CSAR_Deep
# from .legacy.CSAR_R_contrastive import CSAR_R_contrastive
# from .legacy.CSAR_R_Confidence import CSAR_R_Confidence
# from .legacy.CSAR_R_Lasso import CSAR_R_Lasso
# from .legacy.CSAR_Lasso import CSAR_Lasso
# from .legacy.CSAR_STE import CSAR_STE
# from .legacy.CSAR_R_L0 import CSAR_R_L0
# from .legacy.CSAR_R_KD import CSAR_R_KD
# from .legacy.CSAR_R_UBR import CSAR_R_UBR
# from .legacy.CSAR_R_CCBPR import CSAR_R_CCBPR
# from .legacy.CSAR_V_BPR import CSAR_V_BPR
# from .legacy.CSAR_V import CSAR_V
# from .legacy.CSAR_VR import CSAR_VR
# from .legacy.CSAR_IAL import CSAR_IAL
# from .legacy.CSAR_BPR_CE import CSAR_BPR_CE


MODEL_REGISTRY = {
    'mf': MF,
    'lightgcn': LightGCN,
    'most-popular': MostPopular,
    'ItemKNN': ItemKNN,
    'NeuMF': NeuMF,
    'ReRankWrapper': ReRankWrapper,
    'csar-r-bpr': CSAR_R_BPR,
    'acf-nll': ACF_NLL,
    'acf-bpr': ACF_BPR,
    'softplusmf': softplusMF,
    'ease': EASE,
    'protomf': ProtoMF,
    'multivae': MultiVAE,
    'csar' : CSAR,
    'csar-sampled': CSAR_Sampled,
    'csar-r' : CSAR_R,
    'csar-bpr' : CSAR_BPR,
}

def get_model(model_name, config, data_loader):
    """
    모델 이름에 해당하는 모델 클래스의 인스턴스를 생성하여 반환합니다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, data_loader)
