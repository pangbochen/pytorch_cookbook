import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils import data

# Compose is just like nn.Sequential method
transform = T.Compose([
    T.Scale(224), # 缩放图片（Image），保持长宽比不变，最短边为224
    T.CenterCrop(224), # 从图片中间切出224*224的图片
    T.ToTensor(), # 转化为ToTensor，并且归一化为[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化为[-1, +1]，规定均值和标准差，三个通道
])

class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

dataset = DogCat('./dataset/dogcat', transforms=transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)


class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            return None, None

from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    '''(dataset, label)'''
    # filter None dataset
    batch = list(filter(lambda x:x[0] is not None, batch))
    return default_collate(batch)

from torch.utils.data import DataLoader

dataset  = NewDogCat('./dataset/dogcat_wrong', transforms=transform)

dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1)
for batch_datas, batch_labels in dataloader:
    print(batch_datas.size(), batch_labels.size())

import random
# 避免数据损坏造成问题的另一个方法
class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self)-1)
            return self[new_index]