import torch
import torch.nn.functional
from torch import nn, FloatTensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from typing import Iterator, Tuple
import pandas as pd


def ensure_input_bounds(lo: float, hi: float, data: Dataset) -> (float, float):
    if lo is None:
        lo = min((X.min() for (X, _) in data)).item()
    if hi is None:
        hi = max((X.max() for (X, _) in data)).item()
    return lo, hi


def flatten_module(module: nn.Module) -> Iterator[nn.Module]:
    for child in module.children():
        if isinstance(child, nn.Sequential): yield from flatten_module(child)
        else: yield child


def create_qmin_weights_dataframe(qmins: list[torch.Tensor], model: nn.Module) -> pd.DataFrame:
    """
    Creates the dataframe of aligned mutual informations, weights and absolute weights of neighbouring neuron pairs.
    :param qmins: The list of quantized mutual informations between neighbouring neurons,
    as returned by compute_neighbours_qmin.
    :param model: The neural network as a pytorch module.
    :return: The pandas dataframe containing "QMIN", "Weights" and "AbsWgs" "Layers" columns.
    """

    params = list(model.parameters())

    qmins_flat = [item.item() for t in qmins for item in t.flatten()]
    params_flat = [item.item() for t in params[::2] for item in t.flatten()]
    params_abs_flat = [item.item() for t in params[::2] for item in t.flatten().abs()]
    layers_flat = [layer_label for i, t in enumerate(qmins) for layer_label in [i] * (t.size()[0] * t.size()[1])]
    #layers_flat = [layer_label for i, t in enumerate(qmins) for layer_label in [f"{i}-{i+1}"] * (t.size()[0] * t.size()[1])]

    return pd.DataFrame(list(zip(qmins_flat, params_flat, params_abs_flat, layers_flat)),
                        columns=["QMIN", "Weights", "AbsWgs", "Layers"])


def compute_qmin(network: nn.Module, data: Dataset, q: int = 2, layeroffset: int = 1,
                 lo: float = None, hi: float = None,) -> list[torch.Tensor]:
    """
    Computes list of Tensors of Quantized mutual information
    between same-layer neurons in respective layers.
    :param network: The neural network, as a Pytorch module.
    :param data: The input dataset.
    :param q: The number of bins to quantize activations into, 2 by default.
    :param layeroffset: Layers of what distance to compare, 1 by default.
    :param lo: Optional, the lower bound on activations input neurons.
    If not provided, minimum of inputs throughout all components of all input instances is used.
    :param hi: Optional, the upper bound on activations input neurons.
    If not provided, maximum of inputs throughout all components of all input instances is used.
    :return: The list of length (layer_count). Each component of the list is a square matrix
    of floats shaped: (layer_size[i] x layer_size[i]), containing the quantized mutual informations.
    """

    lo, hi = ensure_input_bounds(lo, hi, data)

    def quantize(module: nn.Module, value: FloatTensor) -> FloatTensor:
        if module is None: return q * (value - lo) / (hi - lo)
        elif isinstance(module, nn.Sigmoid): return q * value
        elif isinstance(module, nn.ReLU): return value.ceil()
        else: raise TypeError(f"Modules of {type(module)} type are not allowed for quantization!")

    def iterate_layers(x: FloatTensor) -> Iterator[Tuple[nn.Module, FloatTensor]]:
        module_last = None
        for module in flatten_module(network):
            if isinstance(module, nn.Linear): yield module_last, x
            if not isinstance(module, nn.Flatten): module_last = module
            x = module(x)
        yield module_last, x

    # TODO VB change here, the rest should work as is.
    # Consider the optimization of only computing a half of the matrix.
    input = torch.stack([X for X, _ in data])
    onehots = [one_hot(quantize(*ma).to(torch.int64).clamp(0,q-1)) for ma in iterate_layers(input)]
    joint = [torch.tensordot(A,B,([0],[0])).div(len(input)) for A,B in zip(onehots,onehots[layeroffset:])]
    return [(((j / j.sum(1, keepdim=True)) / j.sum(3, keepdim=True)).log2().nan_to_num(0, 0, 0) * j).sum((1,3)) for j in joint]
