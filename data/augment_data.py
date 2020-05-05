#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
import pathlib

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
file_dir = pathlib.Path(__file__).parent.absolute()


with open(os.path.join(file_dir, "boxes.json")) as file:
    boxes = json.loads(file.read())


originals = {
    waldo: np.array(
        Image.open(os.path.join(file_dir, "data", "original-images", waldo))
    )
    for waldo in os.listdir(os.path.join(file_dir, "data", "original-images"))
    if waldo.endswith(".jpg") and waldo in boxes
}


def augment_data(augment_times=100):

    data = []

    original_names = list(originals.keys())

    for _ in range(augment_times):
        np.random.shuffle(original_names)
        for name in original_names:
            image = originals[name]
            crop_width = int(image.shape[1] * np.random.uniform(0.75, 0.95))
            crop_height = int(image.shape[0] * np.random.uniform(0.75, 0.95))
            crop_x = np.random.randint(image.shape[1] - crop_width)
            crop_y = np.random.randint(image.shape[0] - crop_height)
            hflip = bool(np.random.randint(2))

            data.append(
                {
                    "name": name,
                    "image": image,
                    "crop": {
                        "x": crop_x,
                        "y": crop_y,
                        "w": crop_width,
                        "h": crop_height,
                        "hflip": hflip,
                    },
                }
            )

            box_x, box_y, box_width, box_height = boxes[name]
            new_box_x = max(box_x - crop_x, 0)
            new_box_y = max(box_y - crop_y, 0)
            new_box_width = min(
                box_width, crop_x + crop_width - box_x, box_x + box_width - crop_x
            )
            new_box_height = min(
                box_height, crop_y + crop_height - box_y, box_y + box_height - crop_y
            )

            if new_box_width > 0 and new_box_height > 0:
                if hflip:
                    new_box_x = crop_width - new_box_x - new_box_width

                data[-1]["box"] = {
                    "x": new_box_x,
                    "y": new_box_y,
                    "w": new_box_width,
                    "h": new_box_height,
                }

    return data
