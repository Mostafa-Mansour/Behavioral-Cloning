import keras
from keras.models import Sequential
from keras.activations import elu
from keras.layers import Dense,Dropout,Flatten,Input,MaxPooling2D,Lambda
from keras.layers.convolutional import Conv2D,Cropping2D,Cropping1D

import numpy as np
import cv2
import matplotlib.pyplot as plt

import csv
import get_data_linux
import get_data_windows

"""
In this piece of code, I import the data that I have recorded before,
the data are in different folders and have been recorded using different operation systems (linux and window),
that is why I use two different functions to get it because of the difference in the way the path is represented 
in different operation systems
"""
x_train_f,y_train_f=get_data_linux.get_data('/home/ros-indigo/Desktop/behavioral_cloning/')
#print(x_train_f.shape)
x_train_b,y_train_b=get_data_linux.get_data('./IMG_b/')
#print(x_train_b.shape)
x_train_w,y_train_w=get_data_windows.get_data('./backward/')
#print(x_train_w.shape)
x_train_w2,y_train_w2=get_data_windows.get_data('./abdo/')
#print(x_train_w2.shape)

""" 
After getting all the data, we need to concatenate it in one big array for features and another one for labels
"""
x_train=np.concatenate((x_train_f,x_train_b,x_train_w,x_train_w2),axis=0)
y_train=np.concatenate((y_train_f,y_train_b,y_train_w,y_train_w2),axis=0)
print(x_train.shape)
#print(y_train.shape)
#exit()

"""
After importing the data, I started to build my model as follows:
- conv (16, (3,3), strides=(1,1), padding='same')
- conv (20, (3,3), strides=(1,1), padding='same')
- conv (24, (5,5), strides=(2,2), padding='same')
- conv (36, (5,5), strides=(2,2))
- conv (48, (5,5), strides=(2,2))
- conv (64, (3,3), strides=(1,1))
- conv (64, (3,3), strides=(1,1))
- Flatten
- Dense(500)
- Dense(100)
- Dense(50)
- Dense(10)
- Dense(1)
"""
model=Sequential()
model.add(Lambda(lambda x:x/127.5 - 1,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='elu',strides=(1,1),padding='same'))
model.add(Conv2D(filters=20,kernel_size=(3,3),activation='elu',strides=(1,1),padding='same'))
model.add(Conv2D(filters=24,kernel_size=(5,5),activation='elu',strides=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),activation='elu',strides=(2,2)))
model.add(Conv2D(filters=48,kernel_size=(5,5),activation='elu',strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='elu',strides=(1,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='elu',strides=(1,1)))
model.add(Flatten())
#model.add(Dropout(0.2))

model.add(Dense(500,activation='elu'))
model.add(Dense(100,activation='elu'))

model.add(Dense(50,activation='elu'))
#model.add(Dropout(0.2))
#model.add(Dropout(0.2))
model.add(Dense(10,activation='elu'))
model.add(Dense(1,activation='tanh'))

"""
After, building the model, I have chosen my optimizer to be 'adam' with a decaying learning rate to overcome the problem of over-fitting,
I tried to use dropout layers (commeted in the code) but I found that the model works better wihtout it.
I have chosen 20% of the data for validation and the model have been trained for 3 epochs
"""

model.compile(optimizer='adam',loss='mse',lr=0.01,decay=0.1)
model.fit(x_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)
model.summary()

model.save('model.h5')



