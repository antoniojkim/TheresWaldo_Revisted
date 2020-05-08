# There's Waldo! (Revisited)

A while ago I worked on a project called [*There's Waldo*](https://github.com/antoniojkim/WheresWaldo). The idea was to use computer vision to attempt to find the famous Waldo on any waldo map. The results were fairly good as the model did a fairly good job in identifying waldo within the maps. However, that project used a very naive sliding window approach to object detection and was thus very slow and overfit significantly on the data.

For this project, I want to revisit the problem of finding Waldo using more modern computer vision approaches explored in the YOLO papers [1, 2, 3, 4, 5].

## [TL;DR](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/final_model.ipynb)

In the end, I concluded that finding Waldo on a *Where's Waldo* map, is an uniquely difficult problem when it comes to object detection. Each map has a unique art style and size. Not to mention that Waldo looks different in every example and comes in many different sizes (usually **very** small). As such, the only way to create a decent model was to provide it with prior knowledge about the maps and where Waldo could be found in each map. With this, the model was able to achieve about 75% mIoU, which does not sound too great, but the results are actually quite good. Plus, the inference time is amazing!

## Experimentation

The following outlines the process I went through to create the model. This includes the methods I tried (most of which evidently did not work).

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

Next, I will iterate by implementing the darknet-19 architecture described in the Yolo paper [2] and train it from scratch using the waldo maps.

On this iteration, I noticed that my model architecture might not be problem here. Even with the new (slightly modified) darknet-19 architecture, trained from scratch, the model is not fairing very well. I now realize that this is the case as most of the label values are zero and the model weights are just simply converging to zero. This is why the loss is going down but the precision is not going up.

After 50 epochs of training, I decided to stop the training and try something else.

### [Model #4: Custom Waldo Head/Darknet-19](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v4.ipynb)

In this iteration, I will stick with the Darknet-19 backbone I implemented in the third iteration, but this time with a different head designed specifically for the Waldo problem. What I propose is to have two separate heads: one for localization and one to compute the bounding boxes.

The results were not very good. After about 25 epochs, I stopped training as I noticed that the mean loss had converged and yet again the mIoU never exceeded zero.

### [Model #5 Part 1: Pretrained Backbone](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v5_pretrain.ipynb)

From the last iteration, I am thinking that the darknet-19 backbone is not doing a great job at extracting features from the waldo map. As such, in this iteration, I will attempt to pretrain the darknet-19 backbone on a classification task first before adding in the object detection head.

It was during the process of attempting to train the classification model that I came to the conclusion that the darknet-19 architecture was not a good fit for the task of extracting features from the waldo maps. What I found was that the model was too complex and did not reduce the kernel size enough before average pooling them into a fixed size feature vector. As a result, the feature vector tended to be very similar, regardless of the input map. As such, I propose a new architecture that I will be henceforth be calling WaldoNet which consists of the following layers:

| Type          | Filters | Size/Stride |
|---------------|---------|-------------|
| Convolutional | 64      | 3x3/1       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 128     | 3x3/1       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 256     | 3x3/2       |
| Convolutional | 128     | 1x1/1       |
| Convolutional | 256     | 3x3/2       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 512     | 3x3/3       |
| Convolutional | 256     | 1x1/1       |
| Convolutional | 512     | 3x3/3       |
|---------------|---------|-------------|
| Convolutional | 500     | 1x1/1       |
| AvgPool       |         | Global      |

The output has shape: `[1, 500]`

Notice that the architecture aggressively reduces the kernel size which is required as the waldo map inputs are fairly large.

After only 15 epochs of training, the model performs fairly well on the tasks of determining if waldo is on the map, as well as which waldo map it is looking at.

### [Model #5 Part 2: Custom Waldo Head](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v5.ipynb)

Using the new WaldoNet architecture, I tried using the custom waldo head I developed in the 4th model iteration.

After 50 epochs, I found the loss values converged and the mIoU never exceeded 8% (though, it is a marked improvement over 0%). To test if the waldo head was appropriate and could work, I tried overfitting the model on test data (only 18 samples). After 500 epochs, the loss values appeared to have converged and the mIoU never exceeded 20%. This indicates that the waldo head may be insufficient.

### [Model #6: Modified WaldoNet/Custom Head](https://github.com/antoniojkim/WheresWaldo-YoloV3/tree/master/model/model_v6.ipynb)

In this next iteration, we use the aforementioned custom Waldo head along with the new modified WaldoNet architecture.

Previously I used a frozen pretrained WaldoNet as the size of the input images would cause my GPU to run out of memory. However, after a ton of trial and error, I found out that this was insufficient. So, I modified the WaldoNet architecture to produce even more aggressively reduced features (hoping that it would still be good enough):

| Type          | Filters | Size/Stride |
|---------------|---------|-------------|
| Convolutional | 64      | 3x3/1       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 128     | 3x3/2       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 256     | 3x3/2       |
| Convolutional | 128     | 1x1/1       |
| Convolutional | 256     | 3x3/2       |
| MaxPool       |         | 2x2/2       |
| Convolutional | 512     | 3x3/2       |
| Convolutional | 256     | 1x1/1       |
| Convolutional | 512     | 3x3/2       |
|---------------|---------|-------------|
| Convolutional | 500     | 1x1/1       |
| AvgPool       |         | Global      |

To my delight, it was. I trained from scratch the WaldoNet with the custom head and it worked... sort of. In the end, I found that the *Where's Waldo* problem is almost uniquely difficult in the world of object detection as each map is wildly different is style and size from any other map. Thus, it was essentially impossible for the model to generalize what waldo looked like and where in the map it could be found. As such, I had to "overfit" the model to recognize what each map looked like and had to give it some prior knowledge about where Waldo would be in the map. From this, after many hundreds of epochs of training, I was able to train the model to about 75% mIoU which it turns out performs fairly well on all the Waldo maps in the dataset.

## Reference

[1] [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

[2] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

[3] [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

[4] [YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/pdf/1910.01271.pdf)

[5] [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

[6] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

This project makes use of the [PyTorch](https://pytorch.org/) Deep Learning framework.
