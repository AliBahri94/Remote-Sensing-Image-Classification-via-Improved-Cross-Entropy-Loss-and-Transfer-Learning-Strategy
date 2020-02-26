"""
Created on Wed Feb 2019

@author: abahri
"""

import numpy as np
import keras
import tensorflow as tf
from keras import backend as K


def ICE(y_true, y_pred):
    
    """
    ICE: Improvised Cross Entropy
    ICE= Penalty_Term + Cross Entropy
    """
    
    _epsilon= K.epsilon()
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    values, indices = tf.nn.top_k(y_pred,2)
    max1= K.max(y_pred, axis=-1)
    ICE_Loss= ((((max1 - K.sum((y_pred*y_true), axis=-1)))) + (-tf.reduce_sum(y_true * K.log(y_pred), axis=-1)))

    return ICE_Loss
