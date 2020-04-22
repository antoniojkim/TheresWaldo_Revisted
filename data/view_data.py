# -*- coding: utf-8 -*-

import os
import json

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image

with open("boxes.json") as file:
    boxes = json.loads(file.read())

waldos = [
    (waldo, np.array(Image.open(os.path.join("data", "original-images", waldo))))
    for waldo in os.listdir(os.path.join("data", "original-images"))
    if waldo in boxes
]
i = 0

fig, ax = plt.subplots(1)


def update_screen():
    waldo, image = waldos[i]
    ax.clear()
    ax.imshow(image)
    x, y, w, h = boxes[waldo]
    ax.add_patch(
        patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
    )
    plt.connect("key_press_event", on_key_pressed)
    plt.show(block=True)


def on_key_pressed(event):
    global i
    if event.key == "left" and i > 0:
        i -= 1
        update_screen()
    elif event.key == "right" and i < len(waldos):
        i += 1
        update_screen()


update_screen()
