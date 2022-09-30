# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import torch
from torch.nn import AdaptiveAvgPool3d, AdaptiveMaxPool3d, AvgPool3d, BatchNorm1d, BatchNorm2d, BatchNorm3d, MaxPool3d, Module, ReLU, Softmax, SyncBatchNorm
from torch.utils.data import DataLoader


_ACTIVATION_LAYERS = {
    'relu': ReLU,
    'softmax': Softmax
}

_BN_LAYERS = {
    'bn_1D': BatchNorm1d,
    'bn_2D': BatchNorm2d,
    'bn_3D': BatchNorm3d,
    'sync_bn': SyncBatchNorm
}

_POOL_LAYERS = {
    'adaptive_avg_pool_3d': AdaptiveAvgPool3d,
    'adaptive_max_pool_3d': AdaptiveMaxPool3d,
    'avg_pool_3d': AvgPool3d,
    'max_pool_3d': MaxPool3d
}

def extract_features(model: Module, loader: DataLoader):
    x, y = [], []
    for x_i, y_i in iter(loader):
        x.append(model(x_i))
        y.append(y_i)
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y
