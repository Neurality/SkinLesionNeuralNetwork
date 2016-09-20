import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
import keras.utils.np_utils as kutils
from external_methods import *

data_path = '../input/PH2Dataset/PH2 Dataset images/'
imgcols = 80
imgrows = 64
aspect_ratio = float(imgrows)/float(imgcols)
data_dir = "../np_data/"

def create_train_data():
    images_dict = {}
    lesions_dict = {}
    labels_dict = {}
    for root, dirs, files in os.walk(data_path):
        for name in files:
            filepath = os.path.join(root, name)
            #print "file:", name
            number = name[3:6]

            if name.find("lesion") > -1:
                #print "lesione!"
                lesions_dict[number]= filepath
            elif name.find("Label") > -1:
                if number not in labels_dict.keys():
                    labels_dict[number] = {}
                labels_dict[number][name[7:-4]] = filepath
            else:
                images_dict[number] = filepath


    print "images_n:", len(images_dict)
    print "lesions_n:", len(lesions_dict)
    print "labels_n:", len(labels_dict)
    
    df = pd.read_csv("../input/PH2Dataset/PH2_dataset.txt", sep="|")
    df.rename(columns=lambda x: x.strip(), inplace=True)
    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]

    df["Name"] = df["Name"].str[-4:]
    df.Colors = df.Colors.str.split()


    df = df[df.Name > 0]
    df.sort(columns="Name")
    
    mkdir(data_dir)
    
    pigment_ordered = df["Pigment Network"].astype("category")
    pigment_ordered = pigment_ordered.cat.codes
    pigment_ordered = kutils.to_categorical(pigment_ordered)
    np.save(data_dir + "pigment_ordered.npy",pigment_ordered)
    
    dots_ordered = df["Dots/Globules"].astype("category")
    dots_ordered = dots_ordered.cat.codes
    dots_ordered = kutils.to_categorical(dots_ordered)
    print "dots_ordered"
    print dots_ordered.shape
    np.save(data_dir + "globules_ordered.npy",dots_ordered)
    
    streaks_ordered = df["Streaks"].astype("category")
    streaks_ordered = streaks_ordered.cat.codes
    streaks_ordered = kutils.to_categorical(streaks_ordered)
    np.save(data_dir + "streaks_ordered.npy",streaks_ordered)
    
    regression_ordered = df["Regression Areas"].astype("category")
    regression_ordered = regression_ordered.cat.codes
    regression_ordered = kutils.to_categorical(regression_ordered)
    np.save(data_dir + "regression_ordered.npy",regression_ordered)
    
    veil_ordered = df["Blue-Whitish Veil"].astype("category")
    veil_ordered = veil_ordered.cat.codes
    veil_ordered = kutils.to_categorical(veil_ordered)
    np.save(data_dir + "veil_ordered.npy",veil_ordered)
    
    diagnosis_ordered = df["Clinical Diagnosis"].values
    diagnosis_ordered = kutils.to_categorical(diagnosis_ordered)
    np.save(data_dir + "diagnosis_ordered.npy",diagnosis_ordered)
    
    asimmetry = df.Asymmetry.values
    asimmetryY = kutils.to_categorical(asimmetry)
    np.save(data_dir + "asymmetry_ordered.npy",asimmetryY)
    
    color_str_lists = df.Colors.values
    color_str_lists = [map(int, n) for n in [x for x in color_str_lists]]
    color_labels = ["White","Red","Light-Brown","Dark-Brown","Blue-Gray","Black"]
    colors = {"White":[],"Red":[],"Light-Brown":[],"Dark-Brown":[],"Blue-Gray":[],"Black":[]}
    
    for color_index in range(1,7):
        for index in range(df.shape[0]):
            color_list = df.iloc[index].loc["Colors"]

            if color_index in color_list:
                colors[color_labels[color_index-1]].append([1])
            else:
                colors[color_labels[color_index-1]].append([0])
    
    for key in colors.keys():
        np.save(data_dir + "color_"+key+"_ordered.npy",np.array(colors[key]))


    np_images_dict = {}
    np_images_dict_reshaped = {}
    np_images_lesions = {}
    trainX = []
    trainY = {"lesions": [], "labels": {}}
    keys = images_dict.keys()
    keys.sort()

    min_cols = 1000
    min_rows = 1000
    fullsized_images = []
    resized_images = []

    for key in keys:
        img = np.array(mpimg.imread(images_dict[key]))
        #print img.shape
        if img.shape[0] < min_rows:
            min_rows = img.shape[0]
        if img.shape[1] < min_cols:
            min_cols = img.shape[1]
        fullsized_images.append(img)

    #print min_cols, min_rows

    
    min_required_cols = int(min_rows / aspect_ratio)
    #print min_required_cols
    for index,img in enumerate(fullsized_images):
        img_center_y = fullsized_images[index].shape[0]/2
        img_center_x = fullsized_images[index].shape[1]/2
        #print img_center_y-min_rows/2
        #print img_center_y+min_rows/2

        resized_images.append(fullsized_images[index][img_center_y-min_rows/2:img_center_y+min_rows/2,img_center_x-min_required_cols/2:img_center_x+min_required_cols/2])

        #print resized_images[index].shape

        resized_images[index] = cv2.resize(resized_images[index], (imgcols, imgrows)) 

        #print resized_images[index].shape
        #print key, img.shape
        trainX.append(resized_images[index])





    np.save(data_dir+"images_ordered.npy",trainX)
    imgplot = plt.imshow(trainX[2])
    print('Saving to .npy files done.')





def create_test_data():
    pass



    
if __name__ == '__main__':
    create_train_data()
    create_test_data()
