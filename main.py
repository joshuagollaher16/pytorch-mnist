from os.path import exists

import torch

from torch import nn
from torch import optim

import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

dev = None
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train, test = random_split(train_data, [30000, 30000])
train_loader = DataLoader(train, batch_size=32)
test_loader = DataLoader(test, batch_size=32)

# define model
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(p=0.1), # prevent over-fitting
    nn.Linear(64, 10)
).to(device)

if exists("model"):
    model.load_state_dict(torch.load('model'))

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# define loss function
loss = nn.CrossEntropyLoss()

# optimizer does gradient descent with loss function

n_epochs = 12


def train():
    print("Training")
    for epoch in range(n_epochs):
        losses = list()
        for batch in train_loader:
            image, label = batch
            b = image.size(0)
            image = image.view(b, -1).to(device)

            # 1-forward
            l = model(image)

            # 2-compute objective function
            label = label.to(device)
            J = loss(l, label)

            # 3-clean gradients
            model.zero_grad()
            # 4-accumulate partial derivatives of J with respect to parameters
            J.backward()

            # 5-step in opposite direction of gradient
            optimizer.step()

            losses.append(J.item())

        print(f'Epoch {epoch + 1}, training loss: {torch.mean(torch.tensor(losses)):0.2f}')


def test():
    print("Testing")
    for epoch in range(n_epochs):
        losses = list()
        for batch in test_loader:
            image, label = batch
            b = image.size(0)
            image = image.view(b, -1).to(device)

            l = model(image)

            label = label.to(device)
            J = loss(l, label)

            print(f'guessed {torch.argmax(l[0])} / actual {label[0]}')

            losses.append(J.item())

        print(f'Epoch {epoch + 1}, training loss: {torch.mean(torch.tensor(losses)):0.2f}')


IS_TRAINING = False

if IS_TRAINING:
    train()
    torch.save(model.state_dict(), "model")
else:
    test()



