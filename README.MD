## Synopsis

We developed this project mainly to learn more about CNNs. Our goal was to output a diagnosis for a skin defection, given as input an RGB image and some hand crafted labels describing the lesions. This diagnosis could be one among the following: Common Nevus, Atypical Nevus or Melanoma.
The Datatset used for this study was created from this publication: [Teresa Mendon�a, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira. PH� - A dermoscopic image database for research and benchmarking, 35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6610779&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6610779)
and it's freely downloadable [here](http://www.fc.up.pt/addi/ph2%20database.html).

## Installation

For the creation of this project we used Keras (@fchollet THANK YOU) with TensorFlow backend. 

## How to use

Create a folder, name it "input", make sure to download the dataset .rar package linked above and to extract it inside the "input" directory. To make the program work, first you have to run:
```
python data.py
```
this program will read all the files inside all the subfolders in "input/PH2Dataset/" and create the necessary .npy files in the "np_data" folder. 
After that, simply run:
```
python diagnosis.py
```
to start the training of the Neural Network.

## Network Architecture

The network is a simple Convnet with extra features that have been hand crafted in the database merged at the bottom of it. From there it's just a couple of dense layers that try to predict the correct class.

## Future Contributions

The next obvious step is to automate the feature extraction process by training a neural network to predict them starting from the raw RGB image.
This would probably require many more samples to train but we welcome pull requests if you have a good idea.