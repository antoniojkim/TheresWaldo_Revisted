# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision as torchv
from PIL import Image

from augment_data import augment_data, originals, boxes


class Dataset:
    def __init__(self, ntimes=100):
        self.data = augment_data(ntimes)
        self.to_tensor = torchv.transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def get_image(self, index):
        dims = self.data[index]["crop"]
        x, y, w, h = dims["x"], dims["y"], dims["w"], dims["h"]
        image = self.data[index]["image"][y : y + h, x : x + w]
        if dims["hflip"]:
            image = torchv.transforms.functional.hflip(Image.fromarray(image))

        return self.to_tensor(image)

    def get_label(self, index, image):
        if "box" in self.data[index]:
            box = self.data[index]["box"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            label = torch.from_numpy(np.log1p((x + w // 2, y + h // 2, w, h, 1)))
        else:
            label = torch.from_numpy(np.log1p((0, 0, 0, 0, 0)))

        return label

    def __getitem__(self, index):
        image = self.get_image(index)
        return image, self.get_label(index, image)


class TestDataset:
    def __init__(self):
        self.data = [(originals[key], boxes[key]) for key in originals.keys()]
        self.to_tensor = torchv.transforms.ToTensor()

    def get_image(self, index):
        test, label = self.data[index]
        return self.to_tensor(test)

    def get_label(self, index, image):
        test, label = self.data[index]
        return torch.from_numpy(np.array(label))

    def __getitem__(self, index):
        image = self.get_image(index)
        return image, self.get_label(index, image)

    def __len__(self):
        return len(self.data)
