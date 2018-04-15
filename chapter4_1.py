import torch
from torch import nn
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)

# for Sequential format
# format 1
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3,3,3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

# format 2
net2 = nn.Sequential(
    nn.Conv2d(3,3,3),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

# format 3
from collections import OrderedDict
net3 = nn.Sequential(
    OrderedDict([
        ('conv1', nn.Conv2d(3,3,3)),
        ('bn1', nn.BatchNorm2d(3)),
        ('relu1', nn.ReLU())
    ])
)

print('net1: ', net1)
print('net2: ', net2)
print('net3: ', net3)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.list = [nn.Linear(3, 4), nn.ReLU()]
        self.module_list  = nn.ModuleList([nn.Conv2d(3,3,3), nn.ReLU()])

    def forward(self):
        pass
