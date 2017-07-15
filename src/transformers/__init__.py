# from .OptTransformer import OptTransformer
# from .OptScalarTransformer import OptScalarTransformer
from .Node2VecTransformer import Node2VecTransformer
# from .GammaSMTransformer import GammaSMTransformer
# from .BigClamTransformer import BigClamTransformer
from .HistLossTransformer import HistLossTransformer
# from .NMFTransformer import NMFTransformer
# from .SVDTransformer import SVDTransformer
from .DeepWalkTransformer import DeepWalkTransformer

transformers = {
    # 'svd': SVDTransformer,
    # 'nmf': NMFTransformer,
    # 'gamma': GammaSMTransformer,
    # 'bigclam': BigClamTransformer,
    'node2vec': Node2VecTransformer,
    'deepwalk': DeepWalkTransformer,
    'hist': lambda nx_G, name, dim, **kwargs: HistLossTransformer(nx_G, name, d=dim,  stoc=False, **kwargs),
    # 'stoc_hist': lambda nx_G, name, dim, **kwargs: HistLossTransformer(nx_G, name, d=dim, stoc=True, **kwargs),
    # 'opt': lambda nx_G, name, dim, **kwargs: OptTransformer(nx_G, name, d=dim, stoc=False, **kwargs),
    # 'stoc_opt': lambda nx_G, name, dim, **kwargs: OptTransformer(nx_G, name, d=dim, stoc=True, **kwargs),
    # 'optsk': lambda nx_G, name, dim, **kwargs: OptScalarTransformer(nx_G, name, d=dim, stoc=False, **kwargs),
    # 'stoc_optsk': lambda nx_G, name, dim, **kwargs: OptScalarTransformer(nx_G, name, d=dim, stoc=True, **kwargs),
}