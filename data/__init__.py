import torch.utils.data as data
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

class ImagelistData(data.Dataset):
    def __init__(self, root, imagelistname, transform=None, target_transform=None):
        self._one_hot_encoding=False
        self.root = root
        self.classnames = set()
        with open(imagelistname, 'r') as infile:
            self.images = infile.readlines()
        self.images = [x.strip().split(';') for x in self.images]
        self._classnames = np.sort(np.unique([x[1] for x in self.images]))
        self.images = [[x[0], self._transform_target(x[1])] for x in self.images]
        self.transform = transform
        self.target_transform = target_transform

    def _transform_target(self, target):
        if self._one_hot_encoding:
            raise NotImplementedError("It might be implemented but not working properly")
            tensor = torch.zeros([1, len(self.classes)])
            tensor[0, np.where(self.classes == target)] = 1
        else:
            tensor = np.where(self.classes == target)[0][0]
        return tensor

    def __getitem__(self, index):
        impath, target = self.images[index]
        img = Image.open(os.path.join(self.root, impath)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return self._classnames