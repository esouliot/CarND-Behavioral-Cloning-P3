# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Behavioral Cloning** 
[//]: # (Image References)

[image1]: ./examples/apex_left.jpg "left apex"
[image2]: ./examples/apex_right.jpg "right apex"
[image3]: ./examples/bridge_left.jpg "left bridge"
[image4]: ./examples/bridge_right.jpg "right bridge"
[image5]: ./examples/dirt_left.jpg "left dirt"
[image6]: ./examples/dirt_right.jpg "right dirt"
[image7]: ./examples/yellow_left.jpg "left yellow"
[image8]: ./examples/yellow_right.jpg "right yellow"
[image9]: ./examples/posts_right.jpg "posts right"

[image10]: ./examples/apex_left.gif
[image11]: ./examples/apex_right.gif
[image12]: ./examples/bridge_left.gif
[image13]: ./examples/bridge_right.gif
[image14]: ./examples/dirt_left.gif
[image15]: ./examples/dirt_right.gif
[image16]: ./examples/dirt_right_corner.gif
[image17]: ./examples/yellow_left.gif
[image18]: ./examples/yellow_right.gif

### Files Submitted & Code Quality

#### 1.) This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results, which you are currently reading!

#### 2.) Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following code in the Miniconda 3 command line 
```sh
python drive.py model.h5
```

#### And images can be recorded by adding a desination folder as a second argument (in this case, run1)
```sh
python drive.py model.h5 run1
```

#### 3.) The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### The model used in this project is taken from the publication ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316) written by a group of computer vision and autonomous vehicle engineers at the NVIDIA Corporation.

The model, written in Keras v1.2.1, using the Sequential class, consists of the following layers, as seen in model.py lines 97-132
- Normalization, Lambda layer, lambda x: x/255 - 0.5
- Cropping, Removing the top 70 and bottom 25 rows of pixels, Input dimensions: 320 x 160 x 3, Output dimensions: 320 x 65 x 3
- Convolutional Layer, 5 x 5 filter, 2 x 2 stride, 24 output layers, ReLU activation, 
- Convolutional Layer, 5 x 5 filter, 2 x 2 stride, 36 output layers, ReLU activation
- Convolutional Layer, 5 x 5 filter, 2 x 2 stride, 48 output layers, ReLU activation
- Convolutional Layer, 3 x 3 filter, 1 x 1 stride, 64 output layers, ReLU activation
- Convolutional Layer, 3 x 3 filter, 1 x 1 stride, 64 output layers, ReLU activation
- Flatten the values of the final Convolutional Layer
- Fully connected layer, 100 Neurons
- Fully connected layer,  50 Neurons
- Fully connected layer,  10 Neurons
- Fully connected layer,   1 Neuron, Output its value

As in the traffic-sign classification project (found [here](https://github.com/esouliot/CarND-Traffic-Sign-Classifier-Project)), the [Adam Optimizer](https://arxiv.org/abs/1412.6980) was used for stochastic gradient descent, with mean-squared error being used as the loss function (since the output values are continuous in this task).

#### Collection of training data

The data collection for this project can be broken down into three phases

1.) Collecting normal lap driving, staying in the center of the road

- The car was driven for two laps going in the counter-clockwise "forward" direction (as the car faces when starting), and another two laps going in the clockwise "reverse" direction (doing a u-turn at the start and going the "wrong way"). This is to help generalize the network, and to avoid overfitting, since the forward direction consists mostly of left turns.

- In the lap runs, care was taken to remain as close as possible to the center of the track, and to corner as smoothly as possible. Since the drive.py module controls for throttle (~9mph), certain sharp corners were performed at less-than max throttle (<30mph) to ensure consistent steering angles through the corners.

2.) Collecting recovery driving, to teach the network what to do when the car veers off center

- As in the lap runs, recoveries were recorded going in both clockwise and counter-clockwise directions, to help generalize the model to recover from the left and the right sides of the road. 

- To train the model, I drove the car onto a given side of the road, where the car would be stepping on a boundary marker (either a yellow lane line, a red and white apex, a black bridge wall)

![Left apex][image1] ![Right apex][image2] 

![Left bridge][image3] ![Right bridge][image4] 

![Left yellow][image7] ![Right yellow][image8]

- When the car found itself on one of these boundaries, I would turn on the recording, perform a recovery to the center of the lane, and turn off the recording. This was done all around the track where the car might find itself veering off the road if it doesn't turn properly

![Left apex gif][image10] ![Right apex gif][image11]

![Left bridge gif][image12] ![Right bridge gif][image13]

![Left yellow gif][image17] ![Right yellow gif][image18]

3.) Collecting supplementary data on one particularly troublesome corner after the stone bridge

- As you may have noticed, I did not make mention of the fourth type of lane boundary, the dirt border with no marking. This boundary type proved to be of particular difficulty, so it gets its own section. In test runs prior to the model being finalized, the car simply would not stay on the course. It would veer into the poles on the right-hand side, run off the road and over the curb coming off of the bridge, or it would make it all the way to the corner, and then take the initiative to go off-roading (not good!)

![Left dirt][image5] ![Right dirt][image6]

- So, to remedy these unwanted moves, I collected extra data both in cornering as normal, and in path correction.

![Dirt corner][image16] ![Dirt right][image15]

- And in the end, the extra data paid off, with the car taking that corner with ease.

#### Creation of the Training Set & Training Process

- Since the image locations and their respective steering values were recorded automatically by the simulator, in CSV form, we could easily import those values with the data processing library of our choosing (in my case, Pandas). As can be seen in model.py, lines 22-24, some text preprocessing on the path names was necessary before training the images on the Amazon Web Services (AWS) instance, since the data was recorded on a machine running Windows 7, but the AWS Udacity environment uses Linux. 

- After the text preprocessing, the aptly named train_test_split and shuffle classes were imported from Scikit-learn in order to shuffle and perform a training/testing split on the data (or, shuffling and training/validation split to be more precise).

- Next, because the AWS instance does not have enough memory to hold thousands of images in one array, we instead use batching. The Keras Sequential class has a method fit_generator to train on data in batches, and it requires a generator method to feed the data. So, from lines 44 to 91, we define a generator function to load data in batches of 32 images. In reality, it turned out to be 64 images per batch, since I augmented the data by flipping the images and feeding in the negative of the steering angle. Nevertheless, these batches of 64 were processed using an NVIDIA GPU on the AWS instance over three epochs, giving a final validation loss <0.01

- It should also be noted that, while I included code in model.py to include data from the left and right cameras, it was ultimately not needed, as the code gave a model capable of driving the car around the track successfully using only the center images and angles.

- And as shown in run1.mp4, the car completes a little more than a lap without going outside the track. The closest the car reached to doing so was in the right hairpin turn after the dirt corner. But, even then, the car managed to stay close to the apex, not veering off the road. 
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

