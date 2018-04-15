# Optim SGD
# define the LeNet
import torch
from torch import nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x
net = Net()

from torch import optim
# the optim part
optimizer = optim.SGD(params=net.parameters(), lr=0.01)
optimizer.zero_grad() # equal to net.zero_grad()

input = Variable(torch.randn(1, 3, 32, 32))
output = net(input)
output.backward(output)

optimizer.step()


# set different learning rate for different parts of networks
optimizer = optim.SGD([
    {'params':net.features.parameters()}, # 学习率为1e-5
    {'params':net.classifier.parameters(), 'lr':1e-2}
], lr=1e-2)



# futher more
special_layers = nn.ModuleList([net.classifier[0],net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))
base_params = filter(lambda  p: id(p) not in special_layers_params, net.parameters())

optimizer = torch.optim.SGD([
    {'params':base_params},
    {'params':special_layers.parameters(), 'lr':0.01}
], lr=0.001)
