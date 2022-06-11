import torch
from torch import nn
from sklearn import preprocessing
import numpy as np


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(1, 20)
        self.linear2 = torch.nn.Linear(20, 10)
        self.linear3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        y = self.linear(x)
        y = torch.nn.LeakyReLU()(y)
        y = self.linear2(y)
        y = torch.nn.LeakyReLU()(y)
        y = self.linear3(y)
        return y


def test_linear1():
    x = torch.randn(1000, 1)
    y = x ** 3 + 5

    mse = torch.nn.MSELoss()
    linear = Linear()
    sgd = torch.optim.SGD(linear.parameters(), .01)
    for epoch in range(10000):
        sgd.zero_grad()
        loss = mse(linear(x), y)
        loss.backward()
        sgd.step()
        print('current epoch=%d loss is %f' % (epoch, loss))


class RegLinear(nn.Module):
    def __init__(self):
        super(RegLinear, self).__init__()
        self.l1 = nn.Linear(14, 128)
        self.l2 = nn.Linear(128, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.l1(x)
        y = self.act(y)
        y = self.l2(y)
        return y


def test_regLinear():
    x = torch.randn(100, 14)
    print('x: ', x, x.size())
    y = x ** 2 + 1
    y = torch.sum(y, -1, True)
    print('y:', y, y.size())
    reg = RegLinear()
    mse = nn.MSELoss()
    sgd = torch.optim.SGD(reg.parameters(), .01)
    for epoch in range(1000):
        sgd.zero_grad()
        p = reg(x)
        loss = mse(p, y)
        loss.backward()
        sgd.step()
        if epoch % 50 == 0:
            print('loss:', loss.data)


def test_regLinear2():
    x = torch.randn(100, 14)
    y = x ** 2 + 1
    y = torch.sum(y, -1, True)
    w1 = torch.randn(14, 128, requires_grad=True)
    b1 = torch.randn(128, requires_grad=True)
    w2 = torch.randn(128, 1, requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)
    lr = -.003
    mse = torch.nn.MSELoss()
    for epoch in range(1000):
        h = x.mm(w1) + b1
        h = torch.relu(h)
        p = h.mm(w2) + b2
        loss = mse.forward(p, y)
        loss.backward()
        if epoch % 50 == 0:
            print('loss:', loss.data.numpy())
            print(b2.data.numpy())
        w1.data.add_(lr * w1.grad.data)
        w2.data.add_(lr * w2.grad.data)
        b1.data.add_(lr * b1.grad.data)
        b2.data.add_(lr * b2.grad.data)
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
    # print(w1, b1, w2, b2)


if __name__ == '__main__':
    test_regLinear2()
