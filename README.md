# WheresWaldo-YoloV3

A while ago I worked on a project called [*There's Waldo*](https://github.com/antoniojkim/WheresWaldo). The idea was to use computer vision to attempt to find the famous Waldo on any waldo map. The results were fairly good as the model did a fairly good job in identifying waldo within the maps. However, that project used a very naive sliding window approach to object detection and was thus very slow and overfit significantly on the data.

For this project, I want to revisit the problem of finding Waldo using more modern computer vision approaches explored in the YOLO papers [1, 2, 3, 4, 5].

## TL;DR

## Data Preparation



## Experimentation

### [Model #1: Initial Trial](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v1.ipynb)

Before referencing any other material and work done by previous authors, I wanted to run a quick experiment using a fairly basic approach to object detection.

In any given *Where's Waldo* map, Waldo only appears once. This means that in any given image, the model only has to predict at most one bounding box. With this observation, I build a model that would directly predict the bounding box values.

Using Resnet-50 [6] as a backbone, I built a very simple artificial neural network head to train on the augmented data.

After about 50 epochs of training, I stopped training after observing that the model was no longer improving. The final model was very limited and it was unable to find a single Waldo in any of the maps.

### [Model #2: Yolo Head](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v2.ipynb)

For the next iteration, I replace the head of the model with the head used in the Yolo paper [1].

After a number of epochs, it is not clear if this model is performing any better than the previous iteration. Both models are getting an mIoU of 0%. I now suspect this is because ResNet-50 was a bad choice as a backbone.

I originall chose ResNet-50 as the backbone as it was fast and was pretrained. Even though the waldo maps are very different from the images that it was pretrained on, I thought it might still be able to extract some useful features from the maps. I can see now that this is not the case.

### [Model #3: Darknet-19 Backbone](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v3.ipynb)

Next, I will iterate by implementing the darknet-19 architecture described in the Yolo paper [1] and train it from scratch using the waldo maps.

On this iteration, I noticed that my model architecture might not be problem here. Even with the new (slightly modified) darknet-19 architecture, trained from scratch, the model is not fairing very well. Though, the mean loss is converging fairly quickly. I now realize that this is the case as most of the label values are zero and the model weights are just simply converging to zero. This is why the loss is going down but the precision is not going up.

## Reference

[1] [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

[2] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

[3] [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

[4] [YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/pdf/1910.01271.pdf)

[5] [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

[6] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

This project makes use of the [PyTorch](https://pytorch.org/) Deep Learning framework.
