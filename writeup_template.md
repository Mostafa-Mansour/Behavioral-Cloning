#** Behavioral Cloning ** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* 1. Use the simulator to collect data of good driving behavior
* 2. Build, a convolution neural network in Keras that predicts steering angles from images
* 3. Train and validate the model with a training and validation set
* 4. Test that the model successfully drives around track one without leaving the road
* 5. Summarize the results 


[//]: # (Image References)

[image1]: test_images/image1.png "img from three cameras"
[image2]: test_images/image2.png "cropping imgs"
[image3]: test_images/image3.png "normalized Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Use the simulator to collect data of good driving behavior
First, I used the Udacity provided simulator to collect data. As a rule of thumb, the more that data, the better the network will train so, I start to think about how I can augment the data. The first idea was to make some image processing to the collected images such as flipping, shifting and so on. Then I thought to use the fact that the Udacity provided simulator is open source. I thought that it will be a good idea of I ask my friend to drive for me. I will collect data from different people, every one will put its driving skill in my network. For different scenarios like curves, they will act differently but among them will be something common and that exactly what I need for my network to learn. I asked them to download the simulator and to drive in two directions for more that 3 laps for each paying attention to put a can at the center of the road and driving quite and safely and recording the data using three cameras. By doing that I got about 60000 images for my network to train. To upload the images you can use [get_data_linux](get_data_linux.py) if you use linux or [get_data_windows](get_data_windows.py) if you collect the data  using windows. For both cases, make sure to put your "driving_log.csv" file with the "IMG" folder in the same (parent) folder and to pass the FULL path of this parent folder ending will ("/") to your functions. For windows users, they should change these two functions in the way they read and use the path.

#### 2. Build, a convolution neural network in Keras that predicts steering angles from images
##### 2.1. Getting this number of images is good. But, the image itself has some features that will not help the network to train well. Not only that but also it may cause a lot of problems in training step. These features are trees, sky, clouds and ... etc. Mainly, My network is interested it road images from three cameras and the steering angel. So, I cropped the non-interesting part of each image, around 70 pixels from top and 30 pixels from the bottom. also, I normalized the images for better results using [Lambda layer](https://keras.io/layers/core/#lambda) in Keras.  
##### 2.2. An appropriate model architecture has been employed
My model is based, basically, on Nvidia model published[here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with adding one more convolution layer because a good driver means a good observer and by adding one more convolution layer will allow my network to capture more details that are useful in autonomous driving.

* My model is build as follows (model.py lines: 33-50)
|	layer 			|	size 	|	activation 	|
|:------------:		|:---------:|:-------------:|
| convolution		|(20,3,3)	|	elu			|
| convolution 		|(24,5,5)	|	elu			|
| convolution 		|(36,5,5)	|	elu			|
| convolution 		|(48,5,5) 	|	elu			|
| convolution 		|(64,3,3) 	|	elu			|
| convolution 		|(64,3,3)	|	elu			|
| fully connected 	|	500		|	elu			|
| fully connected 	|	100		|	elu			|
| fully connected 	|	50 		|	elu			|
| fully connected 	|	10 		|	elu			|
| output 			|	1		|	tanh		|
 

* The model includes ELU layers to introduce nonlinearity and to overcome the problem of vanishing the gradient.

#### 3. Train and validate the model with a training and validation set
##### 3.1. The model used and adam optimizer with learning rate=0.01 with a decay =0.1 to prevent over fitting (model.py line 51).
##### 3.2. The model was trained for 5 epochs (model.py line 52) using 20% of the collecting data for validation.

#### 4. Test that the model successfully drives around track one without leaving the road
After training the network, the final model was (saved)[model.h5]. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. To have a look at it, please check this [video](run1.mp4).
 
#### 5. Summarize the results

##### 5.1. Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, I asked my friends to collect me more data using the same procedures, keeping the vehicle driving on the road and making a combination of center lane driving and recovering from both sides. Also, we have driven in both direction to make data balancing instead of flipping the image and inverse the steering angle implicitly. I thought changing the driving direction have an advantage on just flipping the image. we collect the data from the center,left and right cameras for the purpose of augmentation. An example of these image can be shown below

![alt text][image1]

As stated before, all the image content can be confused for our network because of trees, sky and other objects that do not affect our decision to steer the wheel with a specific angle in a specific direction. What makes sense is how the road looks like that is why I cropped a part of every image to get only the valuable view, the road sides. An example of such images can be shown below

![alt text][image2]

##### 5.2. Normalizing the input is one of the most important tricks in any convolution neural network. It allows the network to learn better. So, my model started with a normalization layer with values obtained from Nvidia paper. an example of the images before and after normalization can be shown below

![alt text][image3]

After that my network architecture is started with 6 convolution layers and 4 connected layers. These convolution layers allow the network to be more observable and get important details like different road sides ( lane lines, platform, bridge side or without any side ). Using 'ELU' as an activation function will allow the network to introduce the nonlinearity and to overcome the problem of gradient vanishing. At the output layer a "tanh" activation function is used to give normalization from -1 to 1 to the steering wheel. "tanh" activation function gives and advantage over other activation functions for time series network, in our case the output layer can be considered as a time series action because steering a wheel will change over time.
##### 5.3. The model was trained several times with several architectures and several values of the hyper-parameters. I found that training it with 20% validation for 3 epochs and using 'adam' optimizer with a starting learning rate=0.01 and a decay of 0.1 to overcome the problem of over-fitting will give better results. I tired to add dropout layers to prevent over-fitting but I found that it leads to bad results so, I decided not to use dropout to prevent over-fitting but to use a decaying learning rate. After 3 epochs the validation loss was about 0.02.
##### 5.4. The model was tested on track one for more than 4 loops, the car did not leave the road. It did very well at curves and difficult textile areas like the bridge. Give a look at this [video](run1.mp4) to see one of its loops.

---
### Further discussion
The model works well on the track at which it learned. I think Making an abstract network that can work on different tracks requires collecting data from different tracks with different characteristics. This exactly what makes difference between good driver and bad one. The more you drive, the more you get experience from different situations, the better you will be. So, my model needs to train ALOT to be a good driver model.
