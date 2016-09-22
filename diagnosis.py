import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import cv2

from keras.regularizers import l2
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model
import keras.layers.core as core
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.utils.np_utils as kutils

sys.path.insert(0, './utils')
from external_methods import *
from callbacks import FileMonitor,ActivationsGIF,WeightsGIF


input_labels = ["images",
                  "asymmetry",
                  "color_Blue-Gray",
                  "color_Black",
                  "color_Light-Brown",
                  "color_Dark-Brown",
                  "color_Red",
                  "color_White",
                  "globules",
                  "pigment",
                  "regression",
                  "streaks",
                  "veil"]

def data():
    '''
    Data providing function:
    - reads all the files conteined inside "./np_data/" Those are numpy.arrays containing the images previousy read using utils/data.py
    - divides them by label, according to the input_labels list
    - divides the whole dataset in training and validation set, with an 80/20 ratio
    
    '''
       
    K.set_image_dim_ordering("tf")
    print("="*30)
    print('Loading and preprocessing train data...')
    print('='*30)
    data_path = './np_data/'
    
    trainX = {}
    
    for label in input_labels:
        trainX[label] = np.load(data_path+label+"_ordered.npy")
    
    trainY = np.load(data_path+"diagnosis_ordered.npy")
    channels = trainX["images"].shape[3]
    img_cols = trainX["images"].shape[2]
    img_rows = trainX["images"].shape[1]
    nb_train = trainX["images"].shape[0]

    trainX["images"] = trainX["images"].astype('float32')

    
    print "Dataset has ", nb_train, " training images"
    
    shuffled_index = np.arange(nb_train)
    
    np.random.shuffle(shuffled_index)
    
    mean = np.mean(trainX["images"])  # mean for data centering
    std = np.std(trainX["images"])  # std for data normalization
    trainX["images"] -= mean
    trainX["images"] /= std
    
    trainnum = trainX["images"].shape[0]
    trainlen = int(trainnum * 0.8)
    valX = {}
    for label in input_labels:
        trainX[label] = trainX[label][shuffled_index]
        valX[label] = trainX[label][trainlen:]
        trainX[label] = trainX[label][0:trainlen]
    
    trainY = trainY[shuffled_index]    
       
    print ("trainlen: ",trainlen)   
    
    print ("train Images: ",trainX["images"].shape)
    print ("val images: ",valX["images"].shape)
    print ("train globules: ",trainX["globules"].shape)
    
    valY = trainY[trainlen:]
    trainY = trainY[:trainlen]
    
    print ("valY shape : ",valY.shape)
    print ("trainY shape: ",trainY.shape)
    
    return trainX, trainY, valX, valY



def model(trainX, trainY, valX, valY):
    '''
    Model providing function
    
    '''

    model_callbacks = []
    img_rows = 64
    img_cols = 80
    
    smooth = 1.
    
    batch_size = 16 
    
    #passing argument 'test' I only train the model for 1 epoch
    #passing argument 'epochN' (with N as a positive int) I train the model for N epochs
    nb_epoch = 300
    
    try:
        nb_epoch = find_argument("epoch")
    except ValueError:
        pass    
    try:
        find_argument("test")
        nb_epoch = 1
        
    except ValueError:
        pass

    
    act = 'relu'
    base_layer_depth = 32
    lmbda = 0.1 
    l2reg = l2(lmbda)
    dropout = 0.5
    opt = Adam() #Adadelta()
    
    ##transforming optimizer and parameters to string
    
    optstr = str(opt.__class__).split(".")[2][:-2]
    lr = opt.get_config()
    lr = lr['lr']
    optstr = optstr + '_lr-{0:.6g}'.format(lr)
    
    pixel_offset = 2
    ### pixel_offset is converted into percentage compared to the image's pixel size
    pixel_offset_w = pixel_offset/img_cols
    pixel_offset_h = pixel_offset/img_rows
    
    print "inputsize: "+ str(img_rows) + ' ' + str(img_cols)
    print "opt: "+ str(optstr)
    print "dropout: "+ str(dropout)
    print "batch_size: "+ str(batch_size)
    print "lambda l2 : "+ str(lmbda)
    print "pixel_offset : "+ str(pixel_offset) 
    
    
    ################### callbacks ###################
    

    modelDir = 'models/logs_D-{0:.3f}'.format(dropout)+'_o-'+optstr+'_lmd-'+str(lmbda)+'_px-'+str(pixel_offset)
    mkdir(modelDir)
    
    early = EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='auto')
    
    #Callback to save the best epoch and, eventually, overwrite it if outperformed (regarding the same model)
    
    checkpoint_name = modelDir + '/best_model.h5' #.{epoch:02d}-{val_loss:.4f}.h5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    
    #Tensorboard for each result
    
    #tb_callback = TensorBoard(log_dir="./"+modelDir, histogram_freq=0, write_graph=True)
    
    #WeightsGIF and ActivationsGIF
    
    weigthsSave = WeightsGIF(modelDir, 1)
    fileSave = FileMonitor(modelDir)
    #activationsSave = ActivationsGIF(modelDir, 1, trainX[0])
    
    
    #model_callbacks.append(tb_callback)
    model_callbacks.append(checkpoint)
    model_callbacks.append(early)
    model_callbacks.append(weigthsSave)
    model_callbacks.append(fileSave)
    #model_callbacks.append(activationsSave)

    ################### Model and Layers definition ###################
    
    image_input = Input(( img_rows, img_cols,3), name = "images")
    conv1 = Convolution2D(base_layer_depth, 5, 5, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(image_input)
    conv1 = core.Dropout(dropout)(conv1)
    conv1 = Convolution2D(base_layer_depth, 5, 5, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(conv1)
    conv1 = core.Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(base_layer_depth*2, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(pool1)
    conv2 = core.Dropout(dropout)(conv2)
    conv2 = Convolution2D(base_layer_depth*2, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(conv2)
    conv2 = core.Dropout(dropout)(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(base_layer_depth*4, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(pool2)
    conv3 = core.Dropout(dropout)(conv3)
    conv3 = Convolution2D(base_layer_depth*4, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(conv3)
    conv3 = core.Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(base_layer_depth*8, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(pool3)
    conv4 = core.Dropout(dropout)(conv4)
    conv4 = Convolution2D(base_layer_depth*8, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(conv4)
    conv4 = core.Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(base_layer_depth*16, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(pool4)
    conv5 = core.Dropout(dropout)(conv5)
    conv5 = Convolution2D(base_layer_depth*16, 3, 3, activation='relu', border_mode='same',  W_regularizer=l2reg, b_regularizer = l2reg)(conv5)
    conv5 = core.Dropout(dropout)(conv5)
    
    flat = core.Flatten()(conv5)
    dense = core.Dense(256, activation='relu')(flat)
    dense = core.Dense(16, activation='relu')(dense)
    
    #Auxiliary Inputs
    aux_inputs_list = []

    for label in input_labels:
        if not label =="images":
            aux_inputs_list.append(Input((trainX[label].shape[1],), name = label))

    inputs_list = [image_input]
    for element in aux_inputs_list:
        inputs_list.append(element)
        
    merge_list = [dense]+aux_inputs_list

    merge_layer = merge(merge_list, mode='concat', concat_axis=1,name="merging")
    dense_final = core.Dense(128, activation='relu',name="final_1")(merge_layer)
    dense_final = core.Dropout(dropout)(dense_final)
    dense_final = core.Dense(64, activation='relu',name="final_2")(dense_final)
    dense_final = core.Dropout(dropout)(dense_final)
    prediction = core.Dense(trainY.shape[1], activation='softmax',name="output")(dense_final)
    
    
    model = Model(input=inputs_list, output=prediction)
    
    model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    hist = model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
           callbacks=model_callbacks, 
           validation_data=(valX, valY)
           )
    
    ################### metrics reporting ###################
    
    val_loss, val_acc = model.evaluate(valX, valY, verbose=0)
    
    name_file_save = 'final_model'
    keras_model_save(model,modelDir,name_file_save) 
    
    return {'loss': val_loss, 'status': STATUS_OK}
    

if __name__ == '__main__':
    trainX, trainY, valX, valY = data()
    model(trainX, trainY, valX, valY)

