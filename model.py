"""
Created on Wed Feb 2019

@author: abahri
"""
import numpy as np
import keras
import os
from keras.layers import Dense, GlobalAveragePooling2D,Reshape,Permute,multiply,GlobalMaxPooling2D, SeparableConv2D, Conv2D, BatchNormalization
from keras import backend as K
K.set_image_data_format('channels_last')


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x




def model(image_shape):

    model_NASNet= keras.applications.NASNetMobile(input_shape= image_shape, include_top= False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    x = model_NASNet.output
    ## Squeeze and Excitation Block
    x=squeeze_excite_block(x, ratio=6)
    x=keras.layers.MaxPool2D((4,4),strides=(1,1))(x)
    x=GlobalAveragePooling2D()(x)
    x=keras.layers.Dropout(0.4)(x)
    x = Dense(728)(x)
    x=keras.layers.Dropout(0.3)(x)
    x = Dense(256)(x)
    x=keras.layers.Dropout(0.2)(x)
    predictions = Dense(21, activation="softmax")(x)
    model = keras.models.Model(inputs=model_NASNet.input, outputs=predictions)
    return model
