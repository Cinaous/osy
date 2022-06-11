from math import pi
import torch

x = torch.tensor([pi / 3, pi / 6], requires_grad=True)
SGD = torch.optim.SGD([x], 1e-1)
for epoch in range(11):
    if epoch:
        SGD.zero_grad()
        F.backward()
        SGD.step()
    F = -(x.cos()** 2).sum() ** 2
    print(f'epoch is {epoch}, x is {x}, F is {F}')

x = torch.tensor([10.], requires_grad=True)
optim = torch.optim.Adam([x], .1)
for step in range(1000):
    if step:
        optim.zero_grad()
        F.backward()
        optim.step()
    F = x ** 4 + 5 * x**3 + 7
    print(f'step is {step}, x is {x}, F is {F}')