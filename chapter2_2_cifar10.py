import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

import torch

show = ToPILImage() # change Tensor to Image

# define the data precession part for the project
transform = transforms.Compose([
    transforms.ToTensor(), # transform to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalization mean and std
])

# for training dataset
trainset = tv.datasets.CIFAR10(
    root='C:/Users/pangbochen/Documents/data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# for testing dataset
testset = tv.datasets.CIFAR10(
    root='C:/Users/pangbochen/Documents/data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# fetch a single data as the example
(data, label) = trainset[100]

# (data+1)/2

# Data loader is iterable
dataiter = iter(trainloader)
images, labels = dataiter.next()

# import the net

from chapter2_1_net import Net

net = Net()
print(net)

# define loss and the optimizer
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for trainig part
# follwing three parts
# input data
# forward and backpropagation
# update parameters
for epoch in range(2):
    running_loss =0.0
    # enumerate, start from 0
    for i, data in enumerate(trainloader, 0):

        # input data
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # zero grads
        optimizer.zero_grad()

        # forward and backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # update parameters
        optimizer.step()

        # log information
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('{%d}, {%5d} loss: {%.4f}'.format(epoch+1, i+1, running_loss/2000))
            # zero running_loss
            running_loss = 0.0
    print('Finished Training')

# then test the model
dataiter = iter(testloader)
images, labels = dataiter.next()
print('actual label：',  ' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2-0.5)).resize((400, 100))

# calculate the score on each class
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)
print('predict results', ' '.join('%08s'%classes[predicted[j]] for j in range(4)))

# predict on all dataset
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('1000 total accuracy is %d'%(correct/total))

# if use GPU
if torch.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    output = net(Variable(images))
    loss = criterion(output, Variable(labels))