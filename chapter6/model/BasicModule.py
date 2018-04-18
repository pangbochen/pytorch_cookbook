import torch
import time

class BasicModule(torch.nn.Module):
    # for save and load module
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/'+self.module_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name