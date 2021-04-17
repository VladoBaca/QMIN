import torch
from torchvision.transforms import ToTensor

import qmin
import utils
import models.mnist
from utils import used_device
from models.mice import MiceDataset
from models.mice import MiceNeuralNetwork
from datasets.mnist_download import MNIST_local
from models.mnist import MnistSmallNN
import pandas as pd
import matplotlib.pyplot as plt
import time

utils.use_cuda()
utils.use_direct = True


def test_mice(model, q, verbose=True):
    training_data = MiceDataset(csv_file="datasets/mice/train.csv")
    # model = torch.load('model_files/mice.pth').to(used_device())
    # q = 2
    start = time.time()
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.3, verbose)
    end = time.time()
    interval = end - start
    print(f"Computation time: {interval}.")
    return qmin_table, interval


def test_mnist(model, q, verbose=True):
    training_data = MNIST_local(
        root="./datasets/mnist",
        train=True,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
    )
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1., verbose)
    print(qmin_table)
    return qmin_table


def create_analyze_df(qmins, model):
    df = qmin.create_qmin_weights_dataframe(qmins, model)
    df.plot.scatter(x="QMIN", y="AbsWgs")
    plt.show()
    print("Pearson:")
    print(df.corr())
    print("Spearman:")
    print(df.corr("spearman"))
    return df


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
    model = torch.load("model_files/mice.pth").to(used_device())
    qmins_dir, time_dir = test_mice(model, q)
    df_dir = create_analyze_df(qmins_dir, model)

    # use_direct(False)
    utils.use_cpu()
    model = torch.load("model_files/mice.pth").to(used_device())
    qmins_undir, time_undir = test_mice(model, q)
    df_undir = create_analyze_df(qmins_undir, model)

    print(f"Direct is {time_undir/time_dir} times faster.")

    q_dir = df_dir.iloc[:, 0]
    q_undir = df_undir.iloc[:, 0]
    diff = q_undir - q_dir
    print("Experiment finished")


q = 2
utils.use_cpu()

model = torch.load('./model_files/mnist_small.pth').to(used_device())
qmins = test_mnist(model, q, True)
torch.save(qmins, f"./computed_data/qmins_mnist_{q}_00.txt")


# experiment_differeneces()

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


print("DONE")

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
