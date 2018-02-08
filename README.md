# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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
* model0.py containing the script to create and train the model for run 0 
* model1.py containing the script to create and train the model for run 1
* model2.py containing the script to create and train the model for run 2
* drive.py for driving the car in autonomous mode
* model0.h5 containing a trained convolution neural network for run 0
* model0.h5 containing a trained convolution neural network for run 1
* model0.h5 containing a trained convolution neural network for run 2

* README.md summarizing the results, which you are currently reading!

#### 2.) Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the any of the following 3 commands in the Miniconda 3 command line
```sh
python drive.py model0.h5
```

```sh
python drive.py model1.h5
```

```sh
python drive.py model2.h5
```

#### And images can be recorded by adding a desination folder as a second argument
```sh
python drive.py model0.h5 run0
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


#### Remarks on the NVIDIA Model versus Other Convolutional Neural Networks

- On a cursory level, it may seem unnecessary to implement a model such as the NVIDIA model used for this task, as opposed to a model such as LeNet or [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), but considering the context in which these networks were designed, it seems to be the best fit to use the NVIDIA model.

- LeNet and AlexNet, though potentially "lighter-weight", were not necessarily designed with autonomous driving in mind. LeNet was originally designed for the task of classifying hand-written numerals, and AlexNet for object classification. The NVIDIA model, on the other hand, was designed for a real-world driving application not unlike the lap driving of this project. 

- As mentioned in the abstract of the team's article, the NVIDIA network was designed with the purpose of deriving a steering measurement implicitly from road markings with minimal training input. And although this task is only run on a simulator, and not a real vehicle such as in the study, the model is highly compatible for this project, as it successfully predicted steering angles implicitly, using boundary markings.

#### Collection of Training Data

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

- Next, because the AWS instance does not have enough memory to hold thousands of images in one array, we instead use batching. The Keras Sequential class has a method fit_generator to train on data in batches, and it requires a generator method to feed the data. So, from lines 44 to 91, we define a generator function to load data in batches of 32 images. In reality, it turned out to be 64 images per batch, since I augmented the data by flipping the images and feeding in the negative of the steering angle. Nevertheless, these batches of 64 were processed using an NVIDIA GPU on the AWS instance over three epochs, giving a final validation loss < 0.1

- It should also be noted that, while I included code in model.py to include data from the left and right cameras, it was ultimately not needed, as the code gave a model capable of driving the car around the track successfully using only the center images and angles.

#### Different Model Settings and Remarks on Overfitting

 As can be seen in the project repository, there are three versions of the network, intended to show various stages of overfitting and overtraining. But, as can be seen in all three run videos, if any over or underfitting occurs, it is not severe enough that the car cannot complete the lap run.
 
- Model 0 trains for three epochs, and saves the model parameters from the third epoch to the model0.h5 file. It can be argued that run 0 shows some slight underfitting, with the car swerving very slightly in the straightaways.

- Model 1 trains for five epochs, saving the epoch with the lowest validation loss to the model1.h5 file. This run shows a slight amount of pulling to the left in segments bounded by the yellow lane lines, but those moves are corrected before the car can veer off the road.

- Model 2 trains for 20 epochs, saving the lowest validation error model to the model2.h5 file. As expected, this model produced the lowest training and validation error (loss: 0.0050 - val_loss: 0.0095), but it didn't seem to be indicative of overfitting to the training, and even if it did, the training data largely consisted of smooth lap driving with consistent steering angles through corners. 

- With these three models in mind, the argument can be made that for the purposes of this project (completing autonomous lap runs around the first track), any evidence of overfitting is inconsequential, given that data was collected properly (i.e., data was collected to teach the car what to do for any given scenario on track one). So, even without steps to minimize overfitting, such as including dropout layers, the model did not show any adverse signs of overfitting to the training data. Though, for the purposes of generalizing to a different road setup, such as in track 2, the model did not generalize well. As discussed below.

#### Blind Runs on the Second Track

- To test the adaptability of this model, three runs were recorded on the jungle track (track 2) using the model trained on track 1. As might be expected, the car does not drive nearly as well on this track, as its asphalt is of a different texture, does not have yellow boundaries or red/white apexes, and has a dashed middle lane line. As such, the learning from the first track does not transfer as well. 

- It should be noted, however, that the "overfitted" model 2 managed to drive somewhat well through a portion of the track before running off of the road. And this is with zero training data recorded from this track. It is likely safe to assume that with sufficient training data from track 2, the model should be able to navigate it successfully. 
