import matplotlib as mpl
mpl.use('Agg')
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import cv2
from keras.regularizers import l2
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
import keras.callbacks as callbacks
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import pickle
import keras
import os
import time
#from data import load_train_data, load_test_data
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
import gc
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import fnmatch
import io
import pylab as pl
smooth = 1.



def save_weights_png(model):
    print "HEY!"
    for layer in model.layers:
        weights = layer.get_weights()
        print layer.name
    pl.figure(figsize=(5, 5))
    pl.title('conv1 weights')
    nice_imshow(pl.gca(), make_mosaic(printable, 6, 6), cmap=cm.binary)



class TelegramBot(object):
    """This monitor send messages to a Telegram chat ID with a bot.
    """
    
    def __init__(self, **kwargs):

        self.check_telegram_module()
        api_token = '230100207:AAELpwh6OmjogwmTz6fJ39LQrwVtRysgVBE'
        import telegram
        import telegram_id
        self.bot = telegram.Bot(token=api_token)
        self.chat_id = telegram_id.getid()
        
        #super(object,self).__init__(**kwargs)

        self.can_plot = True

    def check_telegram_module(self):

        try:
            import telegram
        except ImportError:
            raise Exception("You don't have the python-telegram-bot library installed. "
                            "Please install it with `pip install python-telegram-bot -U` ")


    def tg_msg(self, msg):
        
        ret = self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=None)
        return ret

    def tg_image(self, fig):
        
        bf = io.BytesIO()
        fig.savefig(bf, format='png')
        bf.seek(0)

        self.bot.sendPhoto(chat_id=self.chat_id, photo=bf)
    


def find_argument(arg):
    index = [i for i, s in enumerate(sys.argv) if arg in s]
    if index:
        print "trovato", arg
        try:
            return int(sys.argv[index[0]].replace(arg,""))
        except ValueError:
            print "trovato argomento senza integer appeso, accetto lo stesso l'input ma non ritorno niente"
            return None
         
    else:
        raise ValueError

def find_all(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


smooth = 1.


def keras_model_save(model, folder='.', name='model'):
    """
    Saves keras model in a .h5 file and a human-readable json file.

    Keyword arguments:
    model -- Keras model obj
    folder -- the path where to save the model (without ending '/')(default '.')
    name -- The name of the saved model files (default 'model')
    """
    try:
        model.save(folder+'/'+name+'.h5')
        f=open(folder+'/'+name+'.json', 'wb')
        modeljson = model.to_json()
        pickle.dump(modeljson, f)
    except StandardError:
            try:
                f=open(folder+'/'+name+'.json', 'wb')
                modeljson = model.to_json()
                pickle.dump(modeljson, f)
                model.save_weights(folder+'/'+name+'.weights')
                f.close()
            except StandardError:
                random_string = ''.join(random.choice(string.lowercase) for x in range(5))
                name = name + '_' + random_string
                f=open(folder+'/'+name+'.json', 'wb')
                modeljson = model.to_json()
                pickle.dump(modeljson, f)
                model.save_weights(folder+'/'+name+'.weights')
                f.close()

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)    

'''
Prende in ingresso tutte le maschere
ritorna un nparray delle dimansioni (#n_masks x #n_param_ellipse)
'''
def fit_ellipse(masks):
    ellipses = []
    
    k = 0
    for i in range(0,len(masks)):
        mask = masks[0]
        #cv2.imwrite("./input/ellipses/10_1_"+str(k)+"_mask.tif", mask)

        contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        #immagine nera
        temp = np.zeros(mask.shape, dtype=np.uint8)
        #cv2.imwrite("./input/ellipses/10_1_mask_black.tif", temp)

        #disegno i contorni su sfondo nero
        cv2.drawContours(temp, contours, -1, (255,255,255), 3)

        #voglio scrivere quest'immagine che, in teoria, e' nera con 
        #i contorni individuati in bianco.. ma esplode il kernel se usata

        #cv2.imwrite("./input/ellipses/10_1_mask_cont.tif", temp)


        #cv2.imwrite("./input/ellipses/10_1_mask_after.tif", mask)
        for cnt in contours:

            param_ellipse = []
            #print "fitto"
            ellipse = cv2.fitEllipse(cnt)

            #cv2.imwrite("./input/ellipses/10_1_"+str(k)+"_mask.tif", mask)

            #disegna l'ellisse
            black = np.zeros(mask.shape, dtype=np.uint8)
            cv2.ellipse(black, ellipse, (255,255,255), 2)
            #cv2.imwrite("./input/ellipses/10_1_"+str(k)+"_mask_ellipse.tif", black)
            k=k+1


            '''
            Coordinate necessarie/ritornate: 
            (ellipse.center.x,
            ellipse.center.y),
            (ellipse.size.height*2,
            ellipse.size.width*2),
            ellipse.angle
            '''
            #print "ellipse"
            #print ellipse

            ellipses.append(ellipse)

    return np.array(ellipses)

def dice_coef(y_true, y_pred):
    """
        Calculate the dice coefficient of the 2 input Tensors.
        Refence: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

        Keyword arguments:
        y_true -- Keras Tensor containing the ground truth
        y_pred -- Keras Tensor containing the prediction
        """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_coef(y_true, y_pred):
    """
        Calculate the dice coefficient of the 2 input Tensors.
        Refence: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

        Keyword arguments:
        y_true -- Keras Tensor containing the ground truth
        y_pred -- Keras Tensor containing the prediction
        """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def neur_coef(y_true, y_pred):
    """
        Calculate the neurality coefficient of the 2 input Tensors.
        if dice coefficient > 0.5 for a single sample the neuarlity coefficient is 1, it is 0 in all other cases

        Keyword arguments:
        y_true -- Keras Tensor containing the ground truth
        y_pred -- Keras Tensor containing the prediction
        """
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    intersection = y_true * y_pred * 1.0

    int_list = K.sum(intersection,axis=-1)
    y_list = K.sum(y_true,axis=-1) * 1.0
    y_pred_list = K.sum(y_pred,axis=-1) * 1.0

    score = K.mean(K.round((2.0*int_list+smooth) / (y_pred_list+ y_list + smooth)))

    return score

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def save_array_jpg(array,n_images,folder='array_images',name='array_'):
    """
        Save a numpy array as images, using rows as single images.

        Keyword arguments:
        array -- numpy array containing the images with shape (n_images, [1,] width, height)
        n_images -- number of images that need to be saved
        folder -- the path where to save the model (without ending '/')(default '.')
        name -- The name of the saved image files (default 'array_')

        """
    #elimino il canale grigio(unico)
    if (len(array.shape) == 4) and (array.shape[1] == 1):
        print "L'array e' 4 dimensionale e ha solo un canale"
        array = np.reshape(array,(array.shape[0],array.shape[2],array.shape[3]))

        #Normalizzo arrray
        array -=array.min()
        array *= 255.0/(array.max())


        if np.amax(array)<=1:
            #print "IL MASSIMO DELL'array e' MENO di 1"
            #print np.amax(array)
            array = np.multiply(array,255.0)
            #else:
            #print "IL MASSIMO DELL'array e' piu di 1"    
            #print np.amax(array)

            for i in xrange(n_images):

                if i%100 == 0:
                    print('saved jpgs '+ str(i) +"/" + str(n_images))
                    print array[i]

                    cv2.imwrite('./'+folder+'/'+str(i)+'_'+name+'.jpg', array[i], [int(cv2.IMWRITE_JPEG_QUALITY), 80])


def find_mislabeled(true_y,predicted_y,threshold=0.5):
    """
    Returns a numpy array with the indexes of the wrongly segmented images comparing the ground
    truth and the prediction through the neurality_coefficient defined above.
    The threshold for correctness here can be tweaked with the threshold argument

    Keyword arguments:
    y_true -- Keras Tensor containing the ground truth
    y_pred -- Keras Tensor containing the prediction
    threshold -- threshold for correctness of predicted pixels (default '0.5')

    Returns -- np.array
    """
    true_y = true_y.reshape(true_y.shape[0], -1)
    predicted_y = predicted_y.reshape(predicted_y.shape[0], -1)
    intersection = true_y * predicted_y * 1.0
    true_y = np.sum(true_y, 1)
    predicted_y = np.sum(predicted_y, 1)
    intersection = np.sum(intersection, 1)

    sl = np.round((2.0*intersection+1.0) / (true_y+predicted_y+1.0))

    indexes = np.arange(true_y.shape[0])
    indexes = indexes[sl<threshold]

    print "%"*30
    print "there are", str(indexes.shape[0]), "mislabeled images out of", str(true_y.shape[0]) ,"(",str(100*indexes.shape[0]/true_y.shape[0]),"%)"
    print "%"*30
    return indexes

def mkdir(dpath):
    """
    Creates a directory if it doesn't exist.

    Keyword arguments:
    dpath -- the path of the directory to be created (without ending '/')
    """
    if not os.path.exists(dpath):
        os.makedirs(dpath)
        
def data():
    '''
    Data providing function:
    
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    '''
    def preprocess(imgs):
        print("="*30)
        print('preprocessing data...')
        
        imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
        for i in range(imgs.shape[0]):
            imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        
        print('Restituisco un array di forma ', imgs_p.shape)
        print('='*30)
        
        return imgs_p
    '''    
    
   
    trials_todo = 10
    
    img_rows = 64
    img_cols = 80
    
        
    
    print("="*30)
    print('Loading and preprocessing train data...')
    print('='*30)
    imgs_train = np.load('imgs_train.npy')
    imgs_train_mask = np.load('imgs_mask_train.npy')
    
    
    
    
    imgs_test = np.load('imgs_test.npy')
    imgs_id_test = np.load('imgs_id_test.npy')
    
    
    trainX = np.ndarray((imgs_train.shape[0], imgs_train.shape[1], img_rows, img_cols), dtype=np.uint8)
    
    
    for i in range(imgs_train.shape[0]):
        trainX[i, 0] = cv2.resize(imgs_train[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    
    
    #imgs_mask_train = preprocess(imgs_mask_train)
    ##preprocess img_masks
    trainY = np.ndarray((imgs_train_mask.shape[0], imgs_train_mask.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs_train_mask.shape[0]):
        trainY[i, 0] = cv2.resize(imgs_train_mask[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    
    

    trainX = trainX.astype('float32')
    
    
    mean = np.mean(trainX)  # mean for data centering
    std = np.std(trainX)  # std for data normalization

    
    trainX -= mean
    trainX /= std

    
    trainY = trainY.astype('float32')
    trainY /= 255.  # scale masks to [0, 1]
    
    trainnum = trainX.shape[0]
    trainlen = int(trainnum * 0.8)
    print ("trainlen: ",trainlen)
    
    #20% validation, 80% training
    valX = trainX[trainlen:,:]
    trainX = trainX[0:trainlen,:]
    print ("trainlen2: ",trainX.shape)
    
    valY = trainY[trainlen:,:]
    trainY = trainY[:trainlen,:]
    
    
    
    
    #imgs_test = preprocess(imgs_test)
    testX = np.ndarray((imgs_test.shape[0], imgs_test.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs_test.shape[0]):
        testX[i, 0] = cv2.resize(imgs_test[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    
    
    testX = testX.astype('float32')
    testX -= mean
    testX /= std
    
    imgs_train = 0
    imgs_train_mask = 0
    imgs_test = 0
    imgs_id_test = 0
    gc.collect()
    
    return trainX, trainY, valX, valY, testX

def make_mosaic(imgs, nrows, ncols, border=1):
    import numpy.ma as ma
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)