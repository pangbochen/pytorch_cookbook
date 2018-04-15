import torch
from torch import nn

class VeryBigModule(nn.Module):
    def __init__(self):
        super(VeryBigModule, self).__init__()

        # two part gaint parameters
        self.GiantParameter1 = torch.nn.Parameter(torch.randn(10000, 10000)).cuda(0)

        self.GiantParameter2 = torch.nn.Parameter(torch.randn(10000, 10000)).cuda(1)

    def forward(self, x):
        x = self.GiantParameter1.mm(x.cuda(0))
        x = self.GiantParameter2.mm(x.cuda(1))
        return x

# crossentropy with weights
criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,3]))

input = torch.autograd.Variable(torch.randn(4, 2)).cuda()
target = torch.autograd.Variable(torch.Tensor([1, 0, 0, 1])).long().cuda()

#
criterion.cuda()
loss = criterion(input, target)

criterion._buffers


# torch.cuda.device
# torch.set_default_tensot_type

# GPU 0 as default
x = torch.cuda.FloatTensor(2, 3)
y = torch.FloatTensor(2, 3).cuda()

# use GPU 1 as default
with torch.cuda.device(1):
    # set tensor on GPU 1
    a = torch.cuda.FloatTensor(2, 3)

    b = torch.FloatTensor(2, 3).cuda()

    print(a.get_device()==1)

    c = a+b
    print(c.get_device())
    z = x+y
    print(z.get_device())

    d = torch.FloatTensor(2, 3).cuda(0)
    print(d.get_device())