import torch


def himmenlblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = torch.tensor([0., 0.], requires_grad=True)
optim = torch.optim.Adam([x])
for step in range(20001):
    if step:
        optim.zero_grad()
        F.backward()
        optim.step()
    F = -himmenlblau(x)
    if step % 1000 == 0:
        print(f'Current step is {step}, x is {x}, F is {F}')

