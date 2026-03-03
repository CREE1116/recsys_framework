# 모델 이름과 클래스를 매핑하는 레지스트리
# main.py에서 config 파일의 모델 이름을 기반으로 동적으로 클래스를 가져오는 데 사용됩니다.

# General Models - Basic
from .general.mf import MF
from .general.lightgcn import LightGCN
from .general.most_popular import MostPopular
from .general.random_rec import RandomRec
from .general.item_knn import ItemKNN
from .general.neumf import NeuMF
from .general.ACF_NLL import ACF_NLL, ACF_BPR
from .general.softplusmf import softplusMF
from .general.macr import MACR
from .general.pure_svd import PureSVD
from .general.naive_bayes import NaiveBayes
from .general.cooccurrence import CoOccurrence
from .general.multivae import MultiVAE
from .general.mmr import MMR
from .general.gf_cf import GF_CF

# General Models - Linear Autoencoder Family
from .general.ease import EASE
from .general.norm_ease import NormEASE
from .general.rlae import RLAE
from .general.slim import SLIM
from .general.elsa import ELSA
from .general.sansa import SANSA
from .general.infinity_ae import Infinity_AE
from .general.svd_ae import SVD_AE
from .general.svd_ease import SVDEASE
from .general.nc_ease import NCEASE

# General Models - Graph & Others
from .general.ultragcn import UltraGCN
from .general.simgcl import SimGCL
from .general.ials import iALS
from .general.protomf import ProtoMF

# CSAR Models
from .csar.CSAR_Basic import CSAR_Basic
from .csar.CSAR_Minimal import CSAR_Minimal
from .csar.CSAR import CSAR
from .csar.CSAR_BPR import CSAR_BPR
from .csar.MinimalCSAR import MinimalCSAR
from .csar.CSAR_Pure import CSAR_Pure
from .csar.ClosedCSAR import ClosedCSAR

# LIRA Models
from .csar.LIRA import LIRA
from .csar.LightLIRA import LightLIRA
from .csar.PowerLIRA import PowerLIRA
from .csar.LightPowerLIRA import LightPowerLIRA
from .csar.SpectralPowerLIRA import SpectralPowerLIRA
from .csar.TaylorLIRA import TaylorLIRA
from .csar.CGLIRA import CGLIRA
from .csar.ChebyshevLIRA import ChebyshevLIRA
from .csar.ASPIRE import ASPIRE
from .csar.ChebyASPIRE import ChebyASPIRE

MODEL_REGISTRY = {
    # General Models
    'mf': MF,
    'lightgcn': LightGCN,
    'most-popular': MostPopular,
    'ItemKNN': ItemKNN,
    'itemknn': ItemKNN,
    'NeuMF': NeuMF,
    'neumf': NeuMF,
    'random-rec': RandomRec,
    'naive-bayes': NaiveBayes,
    'pure-svd': PureSVD,
    'acf-nll': ACF_NLL,
    'acf-bpr': ACF_BPR,
    'softplusmf': softplusMF,
    'macr': MACR,
    'protomf': ProtoMF,
    'cooccurrence': CoOccurrence,
    'multivae': MultiVAE,
    'mmr': MMR,
    'gf_cf': GF_CF,

    # Linear Autoencoder Family
    'ease': EASE,
    'norm_ease': NormEASE,
    'rlae': RLAE,
    'slim': SLIM,
    'elsa': ELSA,
    'sansa': SANSA,
    'infinity_ae': Infinity_AE,
    'svd_ae': SVD_AE,
    'svd_ease': SVDEASE,
    'nc_ease': NCEASE,

    # Graph & Others
    'ultragcn': UltraGCN,
    'simgcl': SimGCL,
    'ials': iALS,

    # CSAR Models
    'csar_basic': CSAR_Basic,
    'csar_minimal': CSAR_Minimal,
    'csar_bpr': CSAR_BPR,
    'csar': CSAR,
    'csar_pure': CSAR_Pure,
    'minimal_csar': MinimalCSAR,
    'closed_csar': ClosedCSAR,

    # LIRA Models
    'lira': LIRA,
    'light_lira': LightLIRA,
    'power_lira': PowerLIRA,
    'light_power_lira': LightPowerLIRA,
    'spectral_power_lira': SpectralPowerLIRA,
    'taylor_lira': TaylorLIRA,
    'cg_lira': CGLIRA,
    'chebyshev_lira': ChebyshevLIRA,
    'spectral_tikhonov_lira': ASPIRE,
    'aspire': ASPIRE,
    'cheby_aspire': ChebyASPIRE,
}

def get_model(model_name, config, data_loader):
    """
    모델 이름에 해당하는 모델 클래스의 인스턴스를 생성하여 반환합니다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, data_loader)
