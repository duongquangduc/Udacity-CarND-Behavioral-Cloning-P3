# **Behavioral Cloning** 

## Writeup Report
The Behavioral Cloning project is a part of Self-Driving Car Nanodegree Program by Udacity. In this project, I applied deep learning technologies to train a convolutional neural network (CNN) to map raw pixels from a front-facing camera directly to steering commands. For the neural network architecture details, please refer to [NVIDIA paper](https://arxiv.org/abs/1604.07316).

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a writeup report


[//]: # (Image References)

[image1]: ./examples/End-to-end-CNN-Architecture.png "End to End Learning for Self-Driving Car Architecture"
[image2]: ./examples/training-5eps.png "Training with 5 epochs"
[image3]: ./examples/training-10eps.png "Training with 10 epochs"
[image4]: ./examples/early-stopping.png "Early Stopping"

---

### Project Structure

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_10eps.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* Video BehavioralCloning.mp4 /*As this file is large, I could not update it into Github for now. Please see in the Youtube link as in the Results section*/

Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture

First, I played around with several models as suggestions from the course like model from comma.ai or NVIDIA. Finally, I chose model End-to-end Learning for Self-Driving Cars provided by NVIDIA
because it achieved very good result in my experiments. The paper of NVIDIA for this architecture can be found here: https://arxiv.org/abs/1604.07316.

![alt text][image1]

The neural network consists of five convolutional layers, followed by three fully connected layers. The network implemented in Keras as the following:
```
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
```

The model was trained by using Adam optimizer and mean square error as loss function.
```
model.compile(loss='mse', optimizer='adam')
```

### Learning Data and Preprocessing

For this project, I used the data samples provided by Udacity. I also recorded new data for two tracks for future verification and improvements.
The data is splitted into two parts, 80% is used for training and 20% is used for validating.

For preprocessig data, the following steps have been used:
* Data augmentation: flipping the images and steering measurements
* Normalizing data and mean-centering data
* Cropping images in Keras


### Training Process and Strategy

I trained several models with different numbers of epochs (5, 6 10, 15, 20) and observed the training loss and validation loss. In order to avoid overfitting, I used early stopping technique.
I choosed the result where training loss is decreasing while validation loss started increasing.

![alt text][image4]
![alt text][image3]


### Results

The car drive itself in normal road. But it still struggles in some parts of the road or different roads.
Please see my result in [this video](https://www.youtube.com/watch?v=YwTNOnwVOt8).

[![Alt text](http://img.youtube.com/vi/YwTNOnwVOt8/0.jpg)](https://www.youtube.com/watch?v=YwTNOnwVOt8)

### Future Works and Improvements

Although the trained model can be used to drive the car pretty good, there are still many things that I can improve this work and take it to higher levels.
I consider the following things for my future works in this project:
* For the End-to-End Learning model, try with further feature engineering techniques and different data samples, applying more advanced techniques to improve the accuracy 
and reduce overfitting like dropout, L2-Regularization, etc...
* Trying with other neural networks like LeNet, VGG, Comma.ai, and other state-of-the-art models for Self-Driving Car the compare the performance.
