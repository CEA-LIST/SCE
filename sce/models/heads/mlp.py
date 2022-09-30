# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Iterable, Optional, Union

from torch import nn, Tensor
from torch.nn import Module, ReLU, BatchNorm1d, SyncBatchNorm

from sce.models.utils import _ACTIVATION_LAYERS, _BN_LAYERS


class MLPHead(Module):
    """Build a MLP head with optional dropout and normalization.

    Args:
        activation_inplace (bool): Inplace operation for activation layers. Defaults to True.
        activation_layer (Union[str, Module]): Activation layer, if str lookup for the module in _ACTIVATION_LAYERS dictionnary. Defaults to ReLU.
        affine (bool): If True, use affine in normalization layer. Defaults to True.
        bias (bool): If True, use bias in linear layer. If norm_layer, bias set to False. Defaults to True.
        dropout (Union[float, Iterable[float]]): Dropout probability, if 0, no dropout layer. Defaults to 0..
        dropout_inplace (bool): If True, use inplace operation in dropout. Defaults to False.
        hidden_dims (Union[int, Iterable[int]]): dimension of the hidden layers (num_layers - 1). If int, use hidden_dims for all hidden layers. Defaults to 2048.
        input_dim (int): Input dimension for the MLP head. Defaults to 2048.
        norm_layer (Optional[Union[str, Module]]): Normalization layer after the linear layer, if str lookup for the module in _BN_LAYERS dictionnary. Defaults to None.
        num_layers (int): Number of layers (number of hidden layers + 1). Defaults to 2.
        last_bias (bool): If True, use bias in output layer. If last_norm and norm_layer set to False. Defaults to True.
        last_norm (bool): If True, Apply normalization to the last layer if norm_layer. Defaults to False.
        last_affine (bool): If True, use affine in output normalization layer. Defaults to False.
        output_dim (int): Output dimension for the MLP head. Defaults to 128.
        last_init_normal (bool): If True, make normal initialization for last layer. Defaults to False.
        init_mean (float): Mean for the last initialization. Defaults to 0.0.
        init_std (float): STD for the last initialization. Defaults to 0.1.
        zero_bias (bool): If True, put zeros to bias for the last initialization. Defaults to True.

    Raises:
        NotImplementedError: If norm_layer is not supported.
    """

    def __init__(
        self,
        activation_inplace: bool = True,
        activation_layer: Union[str, Module] = ReLU,
        affine: bool = True,
        bias: bool = True,
        dropout: Union[float, Iterable[float]] = 0.,
        dropout_inplace: bool = False,
        hidden_dims: Union[int, Iterable[int]] = 2048,
        input_dim: int = 2048,
        norm_layer: Optional[Union[str, Module]] = None,
        num_layers: int = 2,
        last_bias: bool = True,
        last_norm: bool = False,
        last_affine: bool = False,
        output_dim: int = 128,
        last_init_normal: bool = False,
        init_mean: float = 0.0,
        init_std: float = 0.01,
        zero_bias: bool = True
    ) -> None:
        super().__init__()

        if type(dropout) is float:
            dropout = [dropout] * num_layers

        if type(activation_layer) is str:
            activation_layer = _ACTIVATION_LAYERS[activation_layer]

        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims] * (num_layers - 1)

        if norm_layer is not None:
            norm = True
            if type(norm_layer) is str:
                norm_layer = _BN_LAYERS[norm_layer]
            if norm_layer not in [BatchNorm1d, SyncBatchNorm]:
                raise NotImplementedError(
                    '{norm_layer} not supported in MLPHead')
        else:
            norm = False

        assert len(hidden_dims) == num_layers - 1
        assert len(dropout) == num_layers

        layers = []

        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dims[i-1]
            dim_out = output_dim if i == num_layers - 1 else hidden_dims[i]

            use_norm = True if norm and (i < num_layers - \
                1 or last_norm and i == num_layers - 1) else False
            use_affine = True if affine and i < num_layers - \
                1 or last_affine and i == num_layers - 1 else False
            use_bias = True if (bias and i < num_layers -
                                1 or last_bias and i == num_layers - 1) and not use_norm else False
            use_activation = True if i < num_layers - 1 else False

            if dropout[i] > 0.:
                layers.append(nn.Dropout(
                    p=dropout[i], inplace=dropout_inplace))
            layers.append(nn.Linear(dim_in, dim_out,
                          bias=use_bias and not use_norm))
            
            # init last normal layer
            if i == num_layers - 1 and last_init_normal:
                layers[-1].weight.data.normal_(mean=init_mean, std=init_std)
                if zero_bias and use_bias:
                    layers[-1].bias.data.zero_()

            if use_norm:
                layers.append(norm_layer(
                    num_features=dim_out, affine=use_affine))
            if use_activation:
                layers.append(activation_layer(inplace=activation_inplace))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
