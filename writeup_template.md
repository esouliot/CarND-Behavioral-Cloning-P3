# **Behavioral Cloning** 
[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Files Submitted & Code Quality

#### 1.) This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results, which you are currently reading!

#### 2.) Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
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
- Fully connected layer,   1 Neuron

As in the traffic-sign classification project (found [here](https://github.com/esouliot/CarND-Traffic-Sign-Classifier-Project)), the [Adam Optimizer](https://arxiv.org/abs/1412.6980) was used for stochastic gradient descent, with mean-squared error being used as the loss function (since the output values are continuous in this task).


#### Collection of training data

The data collection for this project can be broken down into three phases

1.) Collecting normal lap driving, staying in the center of the road

- The car was driven for two laps going in the counter-clockwise "forward" direction (as the car faces when starting), and another two laps going in the clockwise "reverse" direction (doing a u-turn at the start and going the "wrong way"). This is to help generalize the network, and to avoid overfitting, since the forward direction consists mostly of left turns.

- In the lap runs, care was taken to remain as close as possible to the center of the track, and to corner as smoothly as possible. Since the drive.py module controls for throttle (~9mph), certain sharp corners were performed at less-than max throttle (<30mph).

2.) Collecting recovery driving, to teach the network what to do when the car veers off center

- As in the lap runs, recoveries were recorded going in both clockwise and counter-clockwise directions, to help generalize the model to recover from the left and the right sides of the road. 

- To train the model, I drove the car onto a given side of the road, where the car would be stepping on a boundary marker (either a yellow lane line, a red and white apex, a black bridge wall, or a dirt border with no markings)

- When the car found itself on one of these four boundaries, I would turn on the recording, perform a recovery to the center of the lane, and turn off the recording. This was done all around the track where the car might find itself veering off the road if it doesn't turn properly

3.) Collecting supplementary data on one particularly troublesome corner after the stone bridge



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
