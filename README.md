# WheresWaldo-YoloV3

A while ago I worked on a project called [*There's Waldo*](https://github.com/antoniojkim/WheresWaldo). The idea was to use computer vision to attempt to find the famous Waldo on any waldo map. The results were fairly good as the model did a fairly good job in identifying waldo within the maps. However, that project used a very naive sliding window approach to object detection and was thus very slow and overfit significantly on the data.

For this project, I want to revisit the problem of finding Waldo using more modern computer vision approaches explored in the YOLO papers [1, 2, 3, 4].

## TL;DR

## Data Preparation



## Experimentation

### [Model #1: Initial Trial](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v1.ipynb)

Before referencing any other material and work done by previous authors, I wanted to run a quick experiment using a fairly basic approach to object detection.

In any given *Where's Waldo* map, Waldo only appears once. This means that in any given image, the model only has to predict at most one bounding box. With this observation, I build a model that would directly predict the bounding box values.

Using Resnet-50 [5] as a backbone, I built a very simple artificial neural network head to train on the augmented data.

After about 50 epochs of training, I stopped training after observing that the model was no longer improving. The final model was very limited and it was unable to find a single Waldo in any of the maps.

### Model #2


## Reference

[1] [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

[2] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

[3] [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

[4] [YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/pdf/1910.01271.pdf)

[5] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

This project makes use of the [PyTorch](https://pytorch.org/) Deep Learning framework.
