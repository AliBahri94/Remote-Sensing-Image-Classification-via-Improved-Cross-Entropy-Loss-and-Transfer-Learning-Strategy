from contextlib import suppress
import matplotlib.pyplot as plt

import numpy as np
import os
import warnings
from zipfile import ZipFile

from skimage.io import imread, imsave

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, classification_report
import keras
from keras.layers import Dense, GlobalAveragePooling2D,Reshape,Permute,multiply,GlobalMaxPooling2D, Conv2D, BatchNormalization
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import load_model
from glob import glob
import cv2
import numpy as np
import keras
from keras.models import load_model


## source direction of dataset
source_dir= "~/dataset name"

## data direction which you want to create
data_dir= "~/" # exp: "~/new dataset"

def devide_dataset(train_percent= 70, valid_percent= 30, source_dir, data_dir):

    np.random.seed(8)
    # Collect class names from directory names in './data/UCMerced_LandUse/Images/'
    class_names = os.listdir(source_dir)    
    
    # Create path to image "flow" base directory
    #flow_base = os.path.join(data_dir, '')
    flow_base = data_dir
    
    # Create pathnames to train/validate/test subdirectories
    target_dirs = {target: os.path.join(flow_base, target) for target in ['train', 'test']}
    
    if not os.path.isdir(flow_base):
    
        # Make new directories
        os.mkdir(flow_base)
        
        for target in ['train', 'test']:
            target_dir = os.path.join(flow_base, target)
            os.mkdir(target_dir)
            for class_name in class_names:
                class_subdir = os.path.join(target_dir, class_name)
                os.mkdir(class_subdir)
    
        # suppress low-contrast warning from skimage.io.imsave
        warnings.simplefilter('ignore', UserWarning)
        
        # Copy images from ./data/UCMerced_LandUse/Images to ./data/flow/<train, validate, test>    
        for root, _, filenames in os.walk(source_dir):
            print(root)
            if filenames:
                class_name = os.path.basename(root)
                count_= len(os.listdir(root))
    
                # Randomly shuffle filenames
                filenames = np.random.permutation(filenames)
                count_train= int((count_*train_percent)/100)
                count_test= count_ - count_train
                for target, count in [('train', count_train), ('test', count_test)]:
                    target_dir = os.path.join(flow_base, target, class_name)
                    for filename in filenames[:count]:
                        filepath = os.path.join(root, filename)
                        image = imread(filepath)
                        basename, _ = os.path.splitext(filename)
                        # Convert TIF to PNG to work with Keras ImageDataGenerator.flow_from_directory
                        target_filename = os.path.join(target_dir, basename + '.png')
                        imsave(target_filename, image)
                
                    filenames = filenames[count:]
        
        # Show future warnings during development
        warnings.resetwarnings()


if __name__ == '__main__':
    
    devide_dataset(train_percent= 70, valid_percent= 30)


