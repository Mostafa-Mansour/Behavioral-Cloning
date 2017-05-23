import keras
from keras.models import Sequential
from keras.activations import elu
from keras.layers import Dense,Dropout,Flatten,Input,MaxPooling2D,Lambda
from keras.layers.convolutional import Conv2D,Cropping2D

import numpy as np
import cv2
import matplotlib.pyplot as plt

import csv
import get_data_linux

import get_data_windows
x_train_f,y_train_f=get_data_linux.get_data('/home/ros-indigo/Desktop/behavioral_cloning/')
#print(x_train_f.shape)
x_train_b,y_train_b=get_data_linux.get_data('./IMG_b/')
#print(x_train_b.shape)
x_train_w,y_train_w=get_data_windows.get_data('./backward/')
#print(x_train_w.shape)
x_train_w2,y_train_w2=get_data_windows.get_data('./abdo/')
#print(x_train_w2.shape)

x_train=np.concatenate((x_train_f,x_train_b,x_train_w,x_train_w2),axis=0)
y_train=np.concatenate((y_train_f,y_train_b,y_train_w,y_train_w2),axis=0)
#print(x_train.shape)
#print(y_train.shape)
#exit()
model=Sequential()
model.add(Lambda(lambda x:x/127.5 - 1,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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

model.compile(optimizer='adam',loss='mse',lr=0.01,decay=0.1)
model.fit(x_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)
model.summary()

model.save('model.h5')



