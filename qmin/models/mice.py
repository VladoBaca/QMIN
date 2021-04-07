import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from qmin import utils
from qmin.utils import used_device

# full = pd.read_csv("../datasets/mice/full.csv")
# train, val = train_test_split(full, test_size=0.15)
# train.to_csv("../datasets/mice/train.csv"), val.to_csv("../datasets/mice/test.csv")


class MiceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.frame.fillna(self.frame.mean(), inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        sample = torch.tensor(self.frame.iloc[idx, 2:79].to_list(), dtype=torch.float)
        label_df = self.frame.iloc[idx, 79:82]
        label_list = [0. if label_df[0] == "Control" else 1.,
                      0. if label_df[1] == "Saline" else 1.,
                      0. if label_df[2] == "S/C" else 1.]
        label = torch.tensor(label_list, dtype=torch.float)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


class MiceNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(77, 20),
            nn.Sigmoid(),
            nn.Linear(20, 10),
            nn.Sigmoid(),
            nn.Linear(10, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        predicted = model(X.to(used_device()))
        loss = loss_fn(predicted, y.to(used_device()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item()/len(X), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model: MiceNeuralNetwork, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            predicted = model(X.to(used_device()))
            y = y.to(used_device())
            test_loss += loss_fn(predicted, y.to(used_device())).item()
            #correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += len(predicted) - (predicted.round() - y).count_nonzero(dim=1).count_nonzero().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def workflow():
    utils.use_cuda()

    batch_size = 64
    learning_rate = 0.5

    train_data = MiceDataset(csv_file="../datasets/mice/train.csv")
    test_data = MiceDataset(csv_file="../datasets/mice/test.csv")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = MiceNeuralNetwork().to(used_device())
    # model = torch.load('../model_files/mice.pth').to(used_device())

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 1000

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Finished training!")

    torch.save(model, '../model_files/mice.pth')

    print("Finished!")
