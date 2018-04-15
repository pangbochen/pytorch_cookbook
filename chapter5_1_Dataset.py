# define own Dataset
# two
# __getitem__
# __len__

import torch
from torch.utils import data

import os
from PIL import Image
import numpy as np

class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        # 所有图片的绝对路径
        # 并不加载，只是在指定路径，调用 __getitem__ 时才会真正读取图片
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_imag = Image.open(img_path)
        array = np.asarray(pil_imag)
        data = torch.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)
dataset = DogCat('./data/dogcat/')
img, label = dataset[0]   # dataset.__getitem__(0)
for img, label in dataset:
    print(img.size(), img.float().mean(), label)