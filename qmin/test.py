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

utils.use_cpu()


def test_mice():
    training_data = MiceDataset(csv_file="datasets/mice/train.csv")
    model = torch.load('model_files/mice.pth').to(used_device())
    q = 2
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.3)
    print(qmin_table)
    return qmin_table


def test_mnist():
    training_data = MNIST_local(
        root="./datasets/mnist",
        train=True,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
    )

    model = torch.load('./model_files/mnist_small.pth').to(used_device())
    q = 2
    qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.)
    print(qmin_table)
    return qmin_table


# qmins = test_mice()
# torch.save(qmins, "./computed_data/qmins_mice_00.txt")

qmins = torch.load("./computed_data/qmins_mice_00.txt")

model = torch.load("model_files/mice.pth").to(used_device())

# params = list(model.parameters())
#
# qmins_flat = [item.item() for t in qmins for item in t.flatten()]
# params_flat = [item.item() for t in params[::2] for item in t.flatten()]
# params_abs_flat = [item.item() for t in params[::2] for item in t.flatten().abs()]
#
#
# df = pd.DataFrame(list(zip(qmins_flat, params_flat, params_abs_flat)),
#                   columns=['QMIN', 'Weights', "AbsWgs"])

df = qmin.create_qmin_weights_dataframe(qmins, model)

print(df)
df.plot.scatter(x="QMIN", y="AbsWgs")
plt.show()
print(df.corr())
print(df.corr("spearman"))


# TODO VB:
#  optimization

print("DONE")