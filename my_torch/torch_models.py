import torch
import tensorflow.keras as kr
import numpy as np
import tensorflow as tf


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (5, 5), padding=2)
        self.conv2 = torch.nn.Conv2d(64, 3, (4, 4), (2, 2), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x


# Net = MyNet()
# optimizer = torch.optim.Adam(Net.parameters())
# loss_fn = torch.nn.MSELoss()
# for step in range(1000):
#     x = torch.rand(17, 3, 32, 32)
#     y = x[..., ::2, ::2]
#     if step:
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     y_pred = Net(x)
#     loss = loss_fn(y_pred, y)
#     print(f'Current step is {step}, loss is {loss}')

Model = kr.Sequential([
    kr.layers.Conv2D(64, 5, padding='same'),
    kr.layers.ReLU(),
    kr.layers.Conv2D(3, 4, 2, padding='same')
])
# Model.compile(optimizer=kr.optimizers.Adam(),
#               loss=kr.losses.mse)
optimizer = kr.optimizers.Adam()
loss_fn = kr.losses.MeanSquaredError()
for epoch in range(1000):
    x = np.random.rand(17, 32, 32, 3)
    y = x[:, ::2, ::2, :]
    with tf.GradientTape() as tape:
        y_pred = Model(x)
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, tape.watched_variables())
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print(f'Current epoch is {epoch}, loss is {loss}')

