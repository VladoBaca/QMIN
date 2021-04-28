import torch
import torch.nn.functional
from torch import nn
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
import math
from typing import Callable, Generator
import pandas as pd

from utils import used_device

quantizable_modules = [nn.Sigmoid, nn.ReLU]
invariant_modules = [nn.Flatten]


def quantize_bounded_input(q: int, bound_low: float, bound_up: float, value: float) -> int:
    return max(min(math.floor((q * (value - bound_low)) / (bound_up - bound_low)), q - 1), 0)


def quantize_sigmoid(q: int, value: float) -> int:
    return min(math.floor(q * value), q - 1)


def quantize_relu(q: int, value: float) -> int:
    return min(math.ceil(value), q - 1)


def flatten_module(module: nn.Module) -> list[nn.Module]:
    modules_flat = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            modules_flat.extend(flatten_module(child))
        else:
            modules_flat.append(child)
    return modules_flat


def iterate_quantized_layers(modules_flat: list[nn.Module], instance: Tensor, quantization_degree: int,
                             input_bound_low: float, input_bound_up: float) -> Generator[torch.LongTensor, None, None]:
    with torch.no_grad():
        activation_tensor = instance.to(used_device())
        module_last = None
        for module in modules_flat:
            if isinstance(module, nn.Linear):
                yield quantize_layer(activation_tensor, module_last, quantization_degree,
                                                       input_bound_low, input_bound_up)
            if not any(isinstance(module, t) for t in invariant_modules):
                module_last = module
            activation_tensor = module(activation_tensor)
        yield quantize_layer(activation_tensor, module_last, quantization_degree, input_bound_low, input_bound_up)


def onehotcounts(data: Dataset, modules_flat: list[nn.Module],
                      quantization_degree: int, input_bound_low: float, input_bound_up: float) -> list[Tensor]:
    xs = tuple(iterate_quantized_layers(modules_flat, X, quantization_degree,input_bound_low, input_bound_up) for X,_ in data)
    return list(one_hot(torch.stack(list(x))) for x in zip(*xs))


def mutual_info(A: Tensor, B: Tensor) -> Tensor:
    return torch.tensordot(A,B,([0],[0]))


def quantize_layer(activation_tensor: torch.FloatTensor, module: nn.Module, quantization_degree: int,
                   input_bound_low: float, input_bound_up: float) -> torch.LongTensor:
    squeezed = activation_tensor.squeeze()
    quantizer = get_quantizer(module, input_bound_low, input_bound_up)
    return torch.LongTensor([quantizer(quantization_degree, activation) for activation in squeezed]).to(used_device())


def get_quantizer(module: nn.Module, bound_low: float = 0., bound_up: float = 0.) -> Callable[[int, float], int]:
    if module is None:
        return lambda q, value: quantize_bounded_input(q, bound_low, bound_up, value)
    elif isinstance(module, nn.Sigmoid):
        return quantize_sigmoid
    elif isinstance(module, nn.ReLU):
        return quantize_relu
    else:
        raise TypeError(f"Modules of {type(module)} type are not allowed for quantization!")


def ensure_input_bounds(input_bound_low: float, input_bound_up: float, data: Dataset) -> (float, float):
    if input_bound_low is None:
        input_bound_low = min((X.min() for (X, _) in data)).item()
    if input_bound_up is None:
        input_bound_up = max((X.max() for (X, _) in data)).item()
    return input_bound_low, input_bound_up


def compute_qmin_tables(direct_count_tables: list[torch.IntTensor], data_len: int,
                        q: int):
    for count in direct_count_tables:
        joint = count.div(data_len)
        marginals = joint.sum(1, keepdim=True) * joint.sum(3, keepdim=True)
        yield (joint / marginals).pow(joint).log2().sum((1,3))


def compute_neighbours_qmin(network: nn.Module, data: Dataset, quantization_degree: int = 2,
                            input_bound_low: float = None, input_bound_up: float = None,
                            verbose: bool = True) -> list[torch.Tensor]:
    """
    Computes list of Tensors of Quantized mutual information
    between neighbouring neurons in respective pairs of layers.
    :param network: The neural network, as a Pytorch module.
    :param data: The input dataset.
    :param quantization_degree: The number of bins to quantize activations into, 2 by default.
    :param input_bound_low: Optional, the lower bound on activations input neurons.
    If not provided, minimum of inputs throughout all components of all input instances is used.
    :param input_bound_up: Optional, the upper bound on activations input neurons.
    If not provided, maximum of inputs throughout all components of all input instances is used.
    :param verbose: True to display progress reports.
    :return: The list of length (layer_count - 1). Each component of the list is a matrix
    of floats shaped: (layer_size[i+1] x layer_size[i]), containing the quantized mutual informations.
    """
    modules_flat = flatten_module(network)

    input_bound_low, input_bound_up = ensure_input_bounds(input_bound_low, input_bound_up, data)

    onehots = onehotcounts(data, modules_flat, quantization_degree, input_bound_low, input_bound_up)

    qmin_tables = list(compute_qmin_tables([mutual_info(A,B) for A, B in zip(onehots, onehots[1:])], len(data), quantization_degree))

    return qmin_tables


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

    df = pd.DataFrame(list(zip(qmins_flat, params_flat, params_abs_flat, layers_flat)),
                      columns=["QMIN", "Weights", "AbsWgs", "Layers"])
    return df


def compute_in_layer_qmin(network: nn.Module, data: Dataset, quantization_degree: int = 2,
                          input_bound_low: float = None, input_bound_up: float = None) -> list[torch.Tensor]:
    """
    Computes list of Tensors of Quantized mutual information
    between same-layer neurons in respective layers.
    :param network: The neural network, as a Pytorch module.
    :param data: The input dataset.
    :param quantization_degree: The number of bins to quantize activations into, 2 by default.
    :param input_bound_low: Optional, the lower bound on activations input neurons.
    If not provided, minimum of inputs throughout all components of all input instances is used.
    :param input_bound_up: Optional, the upper bound on activations input neurons.
    If not provided, maximum of inputs throughout all components of all input instances is used.
    :param verbose: True to display progress reports.
    :return: The list of length (layer_count). Each component of the list is a square matrix
    of floats shaped: (layer_size[i] x layer_size[i]), containing the quantized mutual informations.
    """
    modules_flat = flatten_module(network)

    input_bound_low, input_bound_up = ensure_input_bounds(input_bound_low, input_bound_up, data)

    # TODO VB change here, the rest should work as is.
    # Consider the optimization of only computing a half of the matrix.
    onehots = onehotcounts(data, modules_flat, quantization_degree, input_bound_low, input_bound_up)

    qmin_tables = list(compute_qmin_tables([mutual_info(A,A) for A in onehots], len(data), quantization_degree))

    return qmin_tables
