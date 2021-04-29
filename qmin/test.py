import torch
from torchvision.transforms import ToTensor
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import time
import os.path
from os import path

import qmin
import utils
import models.mnist
from utils import used_device
from models.mice import MiceDataset
from models.mice import MiceNeuralNetwork
from datasets.mnist_download import MNIST_local
from models.mnist import MnistSmallNN


def test_mice(model, q, pairlayers=qmin.neighbours) -> list[torch.Tensor]:
    training_data = MiceDataset(csv_file="datasets/mice/train.csv")
    # model = torch.load('model_files/mice.pth').to(used_device())
    # q = 2
    start = time.time()
    with torch.no_grad():
        qmin_table = qmin.compute_qmin(pairlayers, model, training_data, q)
    end = time.time()
    interval = end - start
    print(f"Computation time: {interval}.")
    return qmin_table


def test_mnist(model, q, verbose=True, pairlayers=qmin.neighbours) -> list[torch.Tensor]:
    training_data = MNIST_local(
        root="./datasets/mnist",
        train=True,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
    )
    start = time.time()
    with torch.no_grad():
        qmin_table = qmin.compute_qmin(pairlayers, model, training_data, q)
    end = time.time()
    interval = end - start
    print(f"Computation time: {interval}.")
    return qmin_table


def analyze_df(df, qmins):
    colors_map = plt.cm.get_cmap("rainbow", len(qmins))

    df.plot.scatter(x="QMIN", y="AbsWgs", s=1, alpha=0.4, c="Layers", cmap=colors_map)
    df.plot.scatter(x="QMIN", y="Weights", s=1, alpha=0.4, c="Layers", cmap=colors_map)
    df_data_only = df.iloc[:, 0:3]

    plt.show()
    print("Pearson:")
    print(df_data_only.corr())
    print()
    print("Spearman rank:")
    print(df_data_only.corr("spearman"))


# def use_direct(direct):
#     utils.use_direct = direct
#     if direct:
#         utils.use_cuda()
#     else:
#         utils.use_cpu()
#     print(f"Using direct: {utils.use_direct}")
#     print(f"Using device: {utils.used_device()}")


def experiment_differeneces():
    q = 2
    # use_direct(True)
    utils.use_cuda()
    model = torch.load("model_files/mice.pth", map_location=torch.device('cpu')).to(used_device())
    qmins_dir = test_mice(model, q)
    df_dir = qmin.create_qmin_weights_dataframe(qmins_dir, model)
    analyze_df(df_dir, qmins_dir)

    # use_direct(False)
    utils.use_cpu()
    model = torch.load("model_files/mice.pth", map_location=torch.device('cpu')).to(used_device())
    qmins_undir = test_mice(model, q)
    df_undir = qmin.create_qmin_weights_dataframe(qmins_undir, model)
    analyze_df(df_undir, qmins_undir)

    q_dir = df_dir.iloc[:, 0]
    q_undir = df_undir.iloc[:, 0]
    diff = q_undir - q_dir
    print("Experiment finished")


dataset_tester_map = {"mice": test_mice, "mnist": test_mnist}
computation_tester_map = {"neighbours": qmin.neighbours, "in_layer": qmin.in_layer}


def load_or_compute_qmins(dataset: str, q: int, model_name: str = None, computation="neighbours") -> (nn.Module, list[torch.Tensor]):
    data_model_part = f"{dataset}" if model_name is None else f"{dataset}_{model_name}"
    model = torch.load(f'./model_files/{data_model_part}.pth', map_location=torch.device("cpu")).to(used_device())

    qmins_path = f"./computed_data/qmins_{computation}_{data_model_part}_{q}_00.txt"
    if path.exists(qmins_path):
        qmins = torch.load(qmins_path)
    else:
        qmins = dataset_tester_map[dataset](model, q, pairlayers=computation_tester_map[computation])
        torch.save(qmins, qmins_path)
    return model, qmins


# model = torch.load('./model_files/mnist_small.pth', map_location=torch.device('cpu')).to(used_device())
# # qmins = torch.load(f"./computed_data/qmins_mnist_{q}_00.txt")
# qmins = test_mnist(model, q, True)
# torch.save(qmins, f"./computed_data/qmins_mnist_{q}_00.txt")


# qmins = test_mnist(model, q, True)
# torch.save(qmins, f"./computed_data/qmins_mnist_{q}_00.txt")

experiment_differeneces()

# use_direct(True)

# model = torch.load("model_files/mice.pth").to(used_device())

# undir is always less...

# torch.save(qmins, f"./computed_data/qmins_mice_{q}_00.txt")
# qmins = torch.load(f"./computed_data/qmins_mice_{q}_00.txt")

# model = torch.load('./model_files/mnist_small.pth').to(used_device())
# qmins = test_mnist(model, q)
# torch.save(qmins, f"./computed_data/qmins_mnist_{q}_00.txt")

# print(df)

# df.plot.scatter(x="QMIN", y="Weights")

# 77-20-10-3  ~ a minute or so

# Mice, q = 2, non-optimized
#              QMIN   Weights    AbsWgs
# QMIN     1.000000 -0.138739  0.554201
# Weights -0.138739  1.000000 -0.040807
# AbsWgs   0.554201 -0.040807  1.000000
#              QMIN   Weights    AbsWgs
# QMIN     1.000000  0.015117  0.512423
# Weights  0.015117  1.000000  0.006323
# AbsWgs   0.512423  0.006323  1.000000

# Mice, q = 2, optimized
#              QMIN   Weights    AbsWgs
# QMIN     1.000000 -0.138739  0.554201
# Weights -0.138739  1.000000 -0.040807
# AbsWgs   0.554201 -0.040807  1.000000
#              QMIN   Weights    AbsWgs
# QMIN     1.000000  0.015117  0.512423
# Weights  0.015117  1.000000  0.006323
# AbsWgs   0.512423  0.006323  1.000000

# Speed:
#   Non-optimized:  20s per 20 instances,  (0.1%  / minute), together ~16.66 hours
#   Optimized:       2s per 20 instances,  (1%    / minute), together  ~1.66 hours
#   Slicing:        <2s per 20 instances,  (1.17% / minute), together  ~1.4  hours
#   Slicing CPU:  <0.5s per 20 instances,  (4.6%  / minute), together  ~0.33 hours
