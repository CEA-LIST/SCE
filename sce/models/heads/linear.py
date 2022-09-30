# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Optional, Union

from torch import Tensor
from torch.nn import Module, Dropout, Linear, Sequential, BatchNorm2d, BatchNorm3d

from sce.models.utils import _BN_LAYERS


class LinearHead(Module):
    """Build a Linear head with optional dropout and normalization.

    Args:
        affine (bool): Use affine in normalization layer. Defaults to True.
        bias (bool): Use bias in linear layer. If norm_layer, bias set to False. Defaults to True.
        dropout (float): Dropout probability, if 0, no dropout layer. Defaults to 0..
        dropout_inplace (bool): Use inplace operation in dropout. Defaults to False.
        input_dim (int): Input dimension for the linear head. Defaults to 2048.
        norm_layer (Optional[Union[str, Module]]): Normalization layer after the linear layer, if str lookup for the module in _BN_LAYERS dictionnary. Defaults to None.
        output_dim (int): Output dimension for the linear head. Defaults to 1000.
        init_normal (bool): If True, make normal initialization for linear layer. Defaults to True.
        init_mean (float): Mean for the initialization. Defaults to 0.0.
        init_std (float): STD for the initialization. Defaults to 0.1.
        zero_bias (bool): If True, put zeros to bias for the initialization. Defaults to True.

    Raises:
        NotImplementedError: If norm_layer is not supported.
    """

    def __init__(
        self,
        affine: bool = True,
        bias: bool = True,
        dropout: float = 0.,
        dropout_inplace: bool = False,
        input_dim: int = 2048,
        norm_layer: Optional[Union[str, Module]] = None,
        output_dim: int = 1000,
        init_normal: bool = True,
        init_mean: float = 0.0,
        init_std: float = 0.01,
        zero_bias: bool = True
    ) -> None:
        super().__init__()

        if norm_layer is not None:
            norm = True
            if type(norm_layer) is str:
                norm_layer = _BN_LAYERS[norm_layer]
            if norm_layer in [BatchNorm2d, BatchNorm3d]:
                raise NotImplementedError(
                    '{norm_layer} not supported in LinearHead')
        else:
            norm = False

        layers = []

        if dropout > 0.:
            layers.append(Dropout(
                p=dropout, inplace=dropout_inplace))

        linear_layer = Linear(input_dim, output_dim, bias=bias and not norm)
        
        # init linear_layer
        if init_normal:
            linear_layer.weight.data.normal_(mean=init_mean, std=init_std)
            if zero_bias and (bias and not norm):
                linear_layer.bias.data.zero_()

        layers.append(linear_layer)

        if norm:
            layers.append(norm_layer(
                num_features=output_dim, affine=affine))

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
