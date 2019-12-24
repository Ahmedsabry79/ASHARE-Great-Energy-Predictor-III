# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:00:07 2019

@author: A-MSabry
"""

import tensorflow as tf

def RMSLE_denorm_loss(y_true, y_pred, means, stds, epsilon = None, *args):
    if epsilon == None:
        epsilon = 0
    y_pred_denormed = (y_pred * (tf.expand_dims(stds, 1) + epsilon)) + tf.expand_dims(means, 1)
    loss = tf.reduce_mean(tf.keras.losses.MSLE(y_true, y_pred_denormed))
    return loss


def RMSE_denorm_loss(y_true, y_pred, means, stds, epsilon = None, *args):
    if epsilon == None:
        epsilon = 0
        
    y_pred_denormed =(y_pred * (tf.expand_dims(stds, 1) + epsilon)) + tf.expand_dims(means, 1)
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred_denormed))

def RMAE_denorm_loss(y_true, y_pred, means, stds, epsilon = None, *args):
    if epsilon == None:
        epsilon = 0
        
    y_pred_denormed = (y_pred * (tf.expand_dims(stds, 1) + epsilon)) + tf.expand_dims(means, 1)
    return tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred_denormed))


def RMSE_loss(y_true, y_pred, *args):    
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

def RMAE_loss(y_true, y_pred, *args):    
    return tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred))