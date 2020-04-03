import torch.utils.data as data
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from torchmeta.utils.data import Dataset

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

    @property
    def num_classes(self):
        return len(self.classes)

class ImagelistMetaDataset(ImagelistData):
    def __init__(self, root, imagelistname, transform=None, target_transform=None):
        super(ImagelistMetaDataset, self).__init__(root, imagelistname, transform, target_transform)
        self._data = [[] for x in range(self.num_classes)]
        for img in self.images:
            self._data[img[1]].append(img[0])
        self.meta_test = False
        self.meta_train = True
        self.meta_val = False
        self.meta_split = None
        self.class_augmentations = []

    def __getitem__(self, index):
        class_name = index % self.num_classes
        data = self._data[class_name]

        return ImagelistMetaData(data, class_name, transform=self.transform,
            target_transform=self.target_transform)

    def __len__(self):
        return self.num_classes


class ImagelistMetaData(Dataset):
    def __init__(self, data, class_name, transform=None, target_transform=None):
        super(ImagelistMetaData, self).__init__(transform=transform, target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)