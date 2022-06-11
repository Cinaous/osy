import torch

sample = torch.rand(10000000, 2)
dist = torch.norm(sample, p=2, dim=1)
dist  = torch.where(dist < 1, 1., 0.)
pi = torch.mean(dist)
pi *= 4
print('pi is {}'.format(pi))

