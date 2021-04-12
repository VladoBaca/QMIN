from datetime import datetime
import torch
from torch import nn
from torch.utils.data import Dataset
import math
from typing import Callable

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


def get_count_tables(shape: list[int], q: int) -> list[torch.IntTensor]:
    count_tables = []
    for i in range(len(shape) - 1):
        count_tables.append(torch.zeros([shape[i], shape[i+1], q, q], dtype=torch.int, device=used_device())
                            .type(torch.IntTensor))
    return count_tables


def fill_count_tables(data: Dataset, modules_flat: list[nn.Module], count_tables: list[torch.IntTensor],
                      quantization_degree: int, input_bound_low: float, input_bound_up: float) -> None:
    for idx, (X, _) in enumerate(data):
        # TODO VB is this necessary?
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
                                                   count_tables[layers_finished - 1])
                    layers_finished += 1
                    quantized_tensor_previous = quantized_tensor_next
                if not any(isinstance(module, t) for t in invariant_modules):
                    module_last = module
                activation_tensor_current = module(activation_tensor_current)
            quantized_tensor_next = quantize_layer(activation_tensor_current, module_last, quantization_degree,
                                                   input_bound_low, input_bound_up)
            if layers_finished > 0:
                fill_counts_for_layer_pair(quantized_tensor_previous, quantized_tensor_next,
                                           count_tables[layers_finished - 1])
            # CPU 20s, cuda ~2:20m (mnist_small)
            if idx % 20 == 0:
                size = len(data)
                print(f"{datetime.now().strftime('%H:%M:%S')}: [{idx:>5d}/{size:>5d}] {100*idx/size:>3.2f}%")


def quantize_layer(activation_tensor: torch.FloatTensor, module: nn.Module, quantization_degree: int,
                   input_bound_low: float, input_bound_up: float) -> torch.ByteTensor:
    squeezed = activation_tensor.squeeze()
    quantizer = get_quantizer(module, input_bound_low, input_bound_up)
    return torch.ByteTensor([quantizer(quantization_degree, activation) for activation in squeezed]).to(used_device())


def get_quantizer(module: nn.Module, bound_low: float = 0., bound_up: float = 0.) -> Callable[[int, float], int]:
    if module is None:
        return lambda q, value: quantize_bounded_input(q, bound_low, bound_up, value)
    elif isinstance(module, nn.Sigmoid):
        return quantize_sigmoid
    elif isinstance(module, nn.ReLU):
        return quantize_relu
    else:
        raise TypeError(f"Modules of {type(module)} type are not allowed for quantization!")


def fill_counts_for_layer_pair(quantized_layer_1: torch.ByteTensor, quantized_layer_2: torch.ByteTensor,
                               count_table: torch.IntTensor) -> None:
    for prev_i, prev_val in enumerate(quantized_layer_1):
        for next_i, next_val in enumerate(quantized_layer_2):
            count_table[prev_i][next_i][prev_val.item()][next_val.item()] += 1


# def get_qmin_tables(shape: list[int]) -> list[torch.FloatTensor]:
#     qmin_tables = []
#     for i in range(len(shape) - 1):
#         qmin_tables.append(torch.zeros([shape[i], shape[i + 1]], dtype=torch.float, device=used_device())
#                            .type(torch.FloatTensor))
#     return qmin_tables

# TODO VB add types!!
def compute_neighbours_qmin(network, data, quantization_degree=2, input_bound_low=None, input_bound_up=None):
    modules_flat = flatten_module(network)
    shape = get_shape(modules_flat)
    count_tables = get_count_tables(shape, quantization_degree)

    if input_bound_low is None:
        input_bound_low = min((X.min() for (X, _) in data)).item()
    if input_bound_up is None:
        input_bound_up = max((X.max() for (X, _) in data)).item()

    fill_count_tables(data, modules_flat, count_tables, quantization_degree, input_bound_low, input_bound_up)

    # TODO VB here finish the mutual information
    # foreach layerpair
    # Divide by sum!!!
    # compute marginals
    # for each cell, compute the term, add to the sum

    qmin_tables = []

    # TODO VB debug this
    # TODO VB either do this on CPU or use some kind of map for the inner computation?
    # or, probably use a sophisticated composition of matrix operations
    for layer_i, count_table in enumerate(count_tables):
        qmin_tables.append(torch.zeros([shape[layer_i], shape[layer_i + 1]], dtype=torch.float, device=used_device())
                           .type(torch.FloatTensor))
        for i, count_table_row in enumerate(count_table):
            for j, confusion_matrix in enumerate(count_table_row):
                marginals_y = confusion_matrix.sum(0).div(len(data))
                marginals_x = confusion_matrix.sum(1).div(len(data))

                mi = 0.0
                for x, confusion_matrix_row in enumerate(confusion_matrix):
                    for y, count_tensor in enumerate(confusion_matrix_row):
                        joint_probability = count_tensor.item() / len(data)
                        # TODO VB use q or 2 for log base?
                        # TODO VB division by zero rethink
                        if joint_probability > 0.0:
                            mi += joint_probability * math.log2(joint_probability /
                                                                (marginals_x[x].item() * marginals_y[y].item()))
                qmin_tables[layer_i][i][j] = mi
    return qmin_tables
