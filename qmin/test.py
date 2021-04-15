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

utils.use_cuda()
utils.use_direct = True


def test_mice(model, q):
    training_data = MiceDataset(csv_file="datasets/mice/train.csv")
    # model = torch.load('model_files/mice.pth').to(used_device())
    # q = 2
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.3)
    print(qmin_table)
    return qmin_table


def test_mnist(model, q):
    training_data = MNIST_local(
        root="./datasets/mnist",
        train=True,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
    )
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.)
    print(qmin_table)
    return qmin_table


print(f"Using direct: {utils.use_direct}")

q = 2
model = torch.load("model_files/mice.pth").to(used_device())
# qmins = test_mice(model, q)
# torch.save(qmins, f"./computed_data/qmins_mice_{q}_00.txt")
qmins = torch.load(f"./computed_data/qmins_mice_{q}_00.txt")

# model = torch.load('./model_files/mnist_small.pth').to(used_device())
# qmins = test_mnist(model, q)
# torch.save(qmins, f"./computed_data/qmins_mnist_{q}_00.txt")

df = qmin.create_qmin_weights_dataframe(qmins, model)
print(df)
# df.plot.scatter(x="QMIN", y="AbsWgs")
df.plot.scatter(x="QMIN", y="Weights")
plt.show()
print(df.corr())
print(df.corr("spearman"))

# TODO VB 
# len(data) !!

# TODO VB:
# use "rows slicing"?

print("DONE")

# 77-20-10-3  ~ a minute or so
#

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
#   Non-optimized: 20s per 20 instances, together ~16.66 hours
#   Optimized:      2s per 20 instance (1% / minute), together ~1.66 hours
