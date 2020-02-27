"""
Created on Wed Feb 2019

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


def evaluate_model():
    ## Data direction
    data_dir= cfg["Dataset_Address_Evaluate"]

    ## Data generator
    image_data_gen_test= ImageDataGenerator(rescale=1./255)
    image_generator_test= image_data_gen_test.flow_from_directory(data_dir, class_mode= cfg["Class_Mode"], target_size= cfg["Target_Size"], batch_size= cfg["Eval_Batch_Size"], shuffle=False)

    ## Predict the model
    
    ## Important Note: if you evaluate your trained model, you have to run following code for loading model
    #NASNET_Model= load_model(cfg["Trained_Model_Path"], custom_objects={ "ICE":ICE})
    
    ## Important Note: if you evaluate our trained model, you have to run following code for loading model
    NASNET_Model= load_model(cfg["Trained_Model_Path"], custom_objects={ "New_loss3":New_loss3})
    
    NASNET_Model.compile(optimizer= keras.optimizers.SGD(lr= cfg["lr"], momentum= cfg["Momentum"]), loss= ICE, metrics=["accuracy"])
    predictions= NASNET_Model.predict_generator(image_generator_test, steps=len(image_generator_test), callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    predictions= np.argmax(predictions, axis=-1)
    for i in range(len(predictions)):
        print("Image %d = %d"%(i, predictions[i]))


if __name__ == '__main__':
    
    evaluate_model()

