"""
Created on Wed Feb 2019

@author: abahri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:25:48 2020

@author: abahri
"""

import os
import warnings
from skimage.io import imread, imsave
from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report
import keras
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import load_model
from glob import glob
import numpy as np
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from model import *
from config import *
from loss_function import *


def train_model():
    
    ## Create model
    model= model(image_shape= cfg["Input_Shape"])
    
    ## Compile the model
    model.compile(optimizer= keras.optimizers.SGD(lr= cfg["lr"], momentum= cfg["Momentum"]), loss= ICE, metrics=["accuracy"])
    
    ## Data direction
    data_dir= cfg["Dataset_Address"]
    target_dirs = {target: os.path.join(data_dir, target) for target in ['train', 'valid']}
    
    ## Data generator
    image_data_gen_train = ImageDataGenerator(rescale=1./255, rotation_range= cfg["Rotation_Range"], width_shift_range= cfg["Width_Shift_Range"], 
                                              height_shift_range= cfg["Height_Shift_Range"], horizontal_flip= cfg["Horizontal_Flip"], 
                                              fill_mode= cfg["Fill_Mode"])
    image_generator_train= image_data_gen_train.flow_from_directory(target_dirs["train"], batch_size= cfg["Train_Batch_Size"], class_mode= cfg["Class_Mode"], target_size= cfg["Target_Size"], shuffle=True)
    image_data_gen_test= ImageDataGenerator(rescale=1./255)
    image_generator_test= image_data_gen_test.flow_from_directory(target_dirs["test"], class_mode= cfg["Class_Mode"], target_size= cfg["Target_Size"], batch_size= cfg["Test_Batch_Size"], shuffle=True)

    ## Checkpoint 
    callback_model= keras.callbacks.ModelCheckpoint( cfg["Save_dir"] + "model.h5", 
                                                     monitor= "val_acc", verbose= 1, save_best_only= True, mode= "max")
    callback_CSV= keras.callbacks.CSVLogger(cfg["Save_dir"] + "model.log", append=True)

    ## Train the model
    model.fit_generator(image_generator_train, steps_per_epoch= len(image_generator_train), epochs= cfg["Epochs"], validation_data= image_generator_test, validation_steps= len(image_generator_test), callbacks=[callback_model, callback_CSV], shuffle= True)        

    

if __name__ == '__main__':
    
    train_model()

     
