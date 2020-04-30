# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision as torchv
from PIL import Image


class Dataset:
    def __init__(self, data):
        self.data = data
        self.to_tensor = torchv.transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dims = self.data[index]["crop"]
        x, y, w, h = dims["x"], dims["y"], dims["w"], dims["h"]
        image = self.data[index]["image"][y : y + h, x : x + w]
        if dims["hflip"]:
            image = torchv.transforms.functional.hflip(Image.fromarray(image))

        train = self.to_tensor(image)

        if "box" in self.data[index]:
            box = self.data[index]["box"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            label = torch.from_numpy(np.log1p((x + w // 2, y + h // 2, w, h, 1)))
        else:
            label = torch.from_numpy(np.log1p((0, 0, 0, 0, 0)))

        return train, label
