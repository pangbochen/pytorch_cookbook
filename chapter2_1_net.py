import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # init as default
        super(Net, self).__init__()
        # equal ti nn.Module.__init__(self)
        # conv net layer1
        # 3 channel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # inchannel, outchannel, kernel_size
        # 5 for 5 * 5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # FC, fullly connected layer
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv -> activate -> pooling
        # (2, 2) for pool
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape '-1' for auto
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)