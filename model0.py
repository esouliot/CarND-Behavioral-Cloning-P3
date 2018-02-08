# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:08:43 2018

@author: jim
"""

import numpy as np
import pandas as pd

import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import ModelCheckpoint

#Importing using Pandas
path = './driving_log.csv'
files = pd.read_csv(path, header = None)

#Had to do a list comprehension to replace backslashes
#Pandas built-in replace methods couldn't pull it off
a = [i.replace('\\', '/') for i in files[0]]
b = [j.replace('\\', '/') for j in files[1]]
c = [k.replace('\\', '/') for k in files[2]]

files[0] = a
files[1] = b
files[2] = c

#Scikit-learn methods
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = files[[0, 1, 2, 3]]
samples = samples.values.tolist()

#Angle offset for left and right images
offset = 0.1

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generator function 
#Taken from Behavioral Cloning Project - Section 17 "Generators"
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center = './IMG/'+batch_sample[0].split('/')[-1]
                #left = './IMG/'+batch_sample[1].split('/')[-1]
                #right = './IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(center)
                #Changing loaded image from BGR to RGB format
                #The autonomous function reads images in RGB
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                """
                left_image = cv2.imread(left)
                left_angle = center_angle + offset
                
                right_image = cv2.imread(right)
                right_angle = center_angle - offset
                """
                center_flip = cv2.flip(src = center_image, flipCode = 0)
                neg_center = -center_angle
                """
                left_flip = cv2.flip(src = left_image, flipCode = 0)
                neg_left = -left_angle
                
                right_flip = cv2.flip(src = right_image, flipCode = 0)
                neg_right = -right_angle
                """
                images.append(center_image)
                #images.append(left_image)
                #images.append(right_image)
                images.append(center_flip)
                #images.append(left_flip)
                #images.append(right_flip)
                
                angles.append(center_angle)
                #angles.append(left_angle)
                #angles.append(right_angle)
                angles.append(neg_center)
                #angles.append(neg_left)
                #angles.append(neg_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()

#Normalize and crop the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0)), input_shape = (3, 160, 320)))

#NVIDIA Convolutional Network
#arXiv:1604.07316v1
model.add(Convolution2D(nb_filter = 24,
                        nb_row = 5,
                        nb_col = 5,
                        subsample = (2, 2),
                        activation = 'relu'))
model.add(Convolution2D(nb_filter = 36,
                        nb_row = 5,
                        nb_col = 5,
                        subsample = (2, 2),
                        activation = 'relu'))
model.add(Convolution2D(nb_filter = 48,
                        nb_row = 5,
                        nb_col = 5,
                        subsample = (2, 2),
                        activation = 'relu'))
model.add(Convolution2D(nb_filter = 64,
                        nb_row = 3,
                        nb_col = 3,
                        activation = 'relu'))
model.add(Convolution2D(nb_filter = 64,
                        nb_row = 3,
                        nb_col = 3,
                        activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit_generator(train_generator, 
                    samples_per_epoch=2*len(train_samples), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=3)
model.save('model0.h5')
