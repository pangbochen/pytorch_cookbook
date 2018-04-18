# use Variable linear regression

import torch
from torch.autograd import Variable

from matplotlib import pyplot as plt
from IPython import display

torch.manual_seed(666)

def get_fake_data(batch_size=8):
    ''' random dataset y = x * 2 + 3 '''
    x = torch.rand(batch_size, 1)*20
    y = x * 2 + (1 + torch.randn(batch_size, 1))*3
    return x, y

x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
#plt.show()


# init parameters
w = torch.rand(1, 1)
b = torch.zeros(1,1)
lr = 0.001

for ii in range(8000):
    x, y = get_fake_data()
    x, y = Variable(x), Variable(y)

    # forward to calculate the loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()

    # back
    loss.backward()

    # update parameter
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # zero grad
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii % 1000 == 0:
        # draw
        display.clear_output(wait=True)
        x = torch.arange(0, 20).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy()) # predicted

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy()) # true dataset

        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.show()
        plt.pause(0.5)

    print(w.data.squeeze()[0], b.data.squeeze()[0])