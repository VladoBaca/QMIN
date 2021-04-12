import torch

import qmin
import utils
from utils import used_device
from models.mice import MiceDataset
from models.mice import MiceNeuralNetwork

utils.use_cpu()

# training_data = MNIST_local(
#     root="datasets/mnist",
#     train=True,
#     transform=ToTensor(),
#     folder="datasets/mnist_local"
# )
# model = torch.load('model_files/mnist_small.pth').to(used_device())

training_data = MiceDataset(csv_file="datasets/mice/train.csv")
model = torch.load('model_files/mice.pth').to(used_device())

q = 2

qmin_table = qmin.compute_neighbours_qmin(model, training_data, q, 0., 1.3)

print(qmin_table)
