import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Grayscale
import glob
from PIL import Image

import utils
from datasets.mnist_download import MNIST_local
from utils import used_device


class MnistNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        results = self.linear_relu_stack(x)
        return results


class MnistSmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        results = self.linear_relu_stack(x)
        return results


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(used_device()))
        loss = loss_fn(pred, y.to(used_device()))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item() / len(X), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(used_device()))
            y = y.to(used_device())
            test_loss += loss_fn(pred, y.to(used_device())).item()
            # correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def workflow():
    utils.use_cuda()

    batch_size = 64
    learning_rate = 1e-2

    #training_data = datasets.MNIST(
    training_data = MNIST_local(
        root="./datasets/mnist",
        train=True,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
        #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=torch.tensor(1.)))
    )

    #test_data = datasets.MNIST(
    test_data = MNIST_local(
        root="./datasets/mnist",
        train=False,
        transform=ToTensor(),
        folder="./datasets/mnist_local"
        #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=torch.tensor(1.)))
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 4, 4
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(test_data), size=(1,)).item()
    #     img, label = test_data[sample_idx]
    #     img = img.transpose(0, 1).transpose(1, 2)
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    model_type = MnistSmallNN
    model_create_new = False
    model_save = True

    model_file_name = "mnist_small.pth" if model_type == MnistSmallNN else "mnist.pth"
    model_file_path = f'./model_files/{model_file_name}'

    model = model_type().to(used_device()) if model_create_new else torch.load(model_file_path).to(used_device())

    # model = MnistNN().to(used_device())
    # model = torch.load('../model_files/mnist.pth').to(used_device())
    # model = MnistSmallNN().to(used_device())
    # model = torch.load('../model_files/mnist_small.pth').to(used_device())

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # loss_fn = nn.MSELoss()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Training finished!")

    if model_save:
        torch.save(model, model_file_path)
        print(f"Saved model to {model_file_path}")

    images = glob.glob(r"./datasets\mnist_my\*png")
    for image in images:
        img = Image.open(image)
        trans0 = ToTensor()
        trans1 = ToPILImage()
        trans2 = Grayscale(num_output_channels=1)

        im = trans2(trans1(trans0(img)))
        # plt.imshow(im)
        tens = trans0(im)
        pred = model(tens.to(used_device()))
        print(image)
        print(pred)
        print(pred.argmax().item())
        print()

    print("Finished!")

# workflow()