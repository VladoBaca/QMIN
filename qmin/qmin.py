from datetime import datetime
import torch
import torch.nn.functional
from torch import nn
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
import math
from typing import Callable
import pandas as pd

import utils
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


def get_shape(modules_flat: list[nn.Module]) -> list[int]:
    last_module = None
    shape = []
    for module in modules_flat:
        if isinstance(module, nn.Linear):
            if not shape:
                shape.append(module.in_features)
                shape.append(module.out_features)
            elif any(isinstance(last_module, t) for t in quantizable_modules):
                if module.in_features != shape[-1]:
                    raise Exception(f"Inconsistent layer sizes: {module.in_features} vs. {shape[-1]}!")
                shape.append(module.out_features)
            else:
                raise TypeError(f"Modules of {type(last_module)} type are not allowed for quantization!")
        if not any(isinstance(module, t) for t in invariant_modules):
            last_module = module
    return shape


def get_count_tables(shape: list[int], q: int) -> list[torch.LongTensor]:
    count_tables = []
    for i in range(len(shape) - 1):
        count_tables.append(torch.zeros([shape[i+1]*q, shape[i]*q], dtype=torch.long, device=used_device()))
    return count_tables


def fill_count_tables(data: Dataset, modules_flat: list[nn.Module], count_tables: list[torch.LongTensor],
                      quantization_degree: int, input_bound_low: float, input_bound_up: float,
                      verbose: bool = False) -> None:
    for idx, (X, _) in enumerate(data):
        with torch.no_grad():
            activation_tensor_current = X.to(used_device())
            module_last = None
            quantized_tensor_previous = None
            layers_finished = 0
            for module in modules_flat:
                if isinstance(module, nn.Linear):
                    quantized_tensor_next = quantize_layer(activation_tensor_current, module_last, quantization_degree,
                                                           input_bound_low, input_bound_up)
                    if layers_finished > 0:
                        fill_counts_for_layer_pair(quantized_tensor_previous, quantized_tensor_next,
                                                   count_tables[layers_finished - 1], quantization_degree)
                    layers_finished += 1
                    quantized_tensor_previous = quantized_tensor_next
                if not any(isinstance(module, t) for t in invariant_modules):
                    module_last = module
                activation_tensor_current = module(activation_tensor_current)
            quantized_tensor_next = quantize_layer(activation_tensor_current, module_last, quantization_degree,
                                                   input_bound_low, input_bound_up)
            if layers_finished > 0:
                fill_counts_for_layer_pair(quantized_tensor_previous, quantized_tensor_next,
                                           count_tables[layers_finished - 1], quantization_degree)
            # CPU 20s, cuda ~2:20m (mnist_small)
            if verbose and idx % 20 == 0:
                size = len(data)
                print(f"{datetime.now().strftime('%H:%M:%S')}: [{idx:>5d}/{size:>5d}] {100*idx/size:>3.2f}%")


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


def fill_counts_for_layer_pair(quantized_layer_1: torch.LongTensor, quantized_layer_2: torch.LongTensor,
                                      count_table: torch.LongTensor, quantization_degree: int) -> None:
    layer_1_one_hot = one_hot(quantized_layer_1, quantization_degree).flatten()

    idx2_tensor = (torch.arange(end=quantized_layer_2.size()[0], device=used_device()) * quantization_degree
                   + quantized_layer_2)

    count_table[idx2_tensor] += layer_1_one_hot


def ensure_input_bounds(input_bound_low: float, input_bound_up: float, data: Dataset) -> (float, float):
    if input_bound_low is None:
        input_bound_low = min((X.min() for (X, _) in data)).item()
    if input_bound_up is None:
        input_bound_up = max((X.max() for (X, _) in data)).item()
    return input_bound_low, input_bound_up


def compute_qmin_tables(direct_count_tables: list[torch.IntTensor], shape: list[int], data_len: int,
                               quantization_degree: int) -> list[torch.Tensor]:
    qmin_tables = []
    for layer_i, direct_count_table in enumerate(direct_count_tables):
        qmin_tables.append(torch.zeros([shape[layer_i + 1], shape[layer_i]], dtype=torch.float, device=used_device())
                           .type(torch.FloatTensor))
        for i in range(shape[layer_i + 1]):
            for j in range(shape[layer_i]):
                confusion_matrix_i = i * quantization_degree
                confusion_matrix_j = j * quantization_degree
                confusion_matrix = direct_count_table[confusion_matrix_i:confusion_matrix_i+quantization_degree,
                                                      confusion_matrix_j:confusion_matrix_j+quantization_degree]

                marginals_y = confusion_matrix.sum(0).div(data_len)
                marginals_x = confusion_matrix.sum(1).div(data_len)

                mi = 0.0
                for x, confusion_matrix_row in enumerate(confusion_matrix):
                    for y, count_tensor in enumerate(confusion_matrix_row):
                        joint_probability = count_tensor.item() / data_len
                        if joint_probability > 0.0:
                            mi += joint_probability * math.log2(joint_probability /
                                                                (marginals_x[x].item() * marginals_y[y].item()))
                qmin_tables[layer_i][i][j] = mi
    return qmin_tables


def compute_neighbours_qmin(network: nn.Module, data: Dataset, quantization_degree: int = 2,
                            input_bound_low: float = None, input_bound_up: float = None,
                            verbose: bool = False) -> list[torch.Tensor]:
    modules_flat = flatten_module(network)
    shape = get_shape(modules_flat)
    count_tables = get_count_tables(shape, quantization_degree)

    input_bound_low, input_bound_up = ensure_input_bounds(input_bound_low, input_bound_up, data)

    fill_count_tables(data, modules_flat, count_tables, quantization_degree, input_bound_low, input_bound_up, verbose)

    qmin_tables = compute_qmin_tables(count_tables, shape, len(data), quantization_degree)

    return qmin_tables


def create_qmin_weights_dataframe(qmins: list[torch.Tensor], model: nn.Module) -> pd.DataFrame:
    params = list(model.parameters())

    qmins_flat = [item.item() for t in qmins for item in t.flatten()]
    params_flat = [item.item() for t in params[::2] for item in t.flatten()]
    params_abs_flat = [item.item() for t in params[::2] for item in t.flatten().abs()]

    df = pd.DataFrame(list(zip(qmins_flat, params_flat, params_abs_flat)),
                      columns=['QMIN', 'Weights', "AbsWgs"])
    return df
