# -*- coding: utf-8 -*-

import os
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

with open("boxes.json") as file:
    boxes = json.loads(file.read())


waldo = next(
    waldo
    for waldo in os.listdir(os.path.join("data", "original-images"))
    if all(
        waldo not in boxes,
        os.path.isfile(os.path.join("data", "original-images", waldo)),
    )
)
image = np.array(Image.open(os.path.join("data", "original-images", waldo)))

x = 0
y = 0
x1 = 0
y1 = 0
w = image.shape[1]
h = image.shape[0]

print(image.shape)


def update_screen():
    print(f'"{waldo}": {[x, y, w, h]}')
    plt.gca()
    plt.imshow(image[y : y + h, x : x + w])
    plt.connect("button_press_event", on_button_pressed)
    plt.connect("button_release_event", on_button_released)
    plt.connect("key_press_event", on_key_pressed)
    plt.show(block=True)


def on_button_pressed(event):
    global x1, y1
    x1 = int(event.xdata if event.xdata is not None else 0)
    y1 = int(event.ydata if event.ydata is not None else 0)


def on_button_released(event):
    global x, y, x1, y1, w, h
    x2 = int(event.xdata if event.xdata is not None else w)
    y2 = int(event.ydata if event.ydata is not None else h)

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    x = x + x1
    y = y + y1

    plt.gcf()
    plt.imshow(image[y : y + h, x : x + w])
    plt.connect("button_press_event", on_button_pressed)
    plt.connect("button_release_event", on_button_released)
    plt.show(block=True)


def on_key_pressed(event):
    global x, y, w, h
    if event.key == "up" and y > 0:
        y -= 1
        update_screen()
    elif event.key == "down" and y < image.shape[0]:
        y += 1
        update_screen()
    elif event.key == "left" and x > 0:
        x -= 1
        update_screen()
    elif event.key == "right" and x < image.shape[1]:
        x += 1
        update_screen()
    elif event.key == "c":
        x = 0
        y = 0
        w = image.shape[1]
        h = image.shape[0]
        update_screen()


update_screen()
