# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 00:18:11 2019

@author: A-MSabry
"""

import tensorflow as tf
import logging
from copy import deepcopy
from Lib.model import model_builder
from Lib.PipeLine import InputPipeLine
import os

def set_logger(logger_name, level = 'info'):
    levels = {"info": logging.INFO,
              "warning": logging.WARNING,
              "debug": logging.DEBUG}
    
    Logger = logging.getLogger(logger_name)
    fileHandler = logging.FileHandler(logger_name+".log")
    
    Logger.setLevel(levels[level])
    Logger.addHandler(fileHandler)
    return Logger


def switch_to_validation_mode(hparams):
    val_params =  deepcopy(hparams)
    val_params.model_mode = 'validation'
    val_params.encoder_keep_prob = 1.
    val_params.enc_dec_keep_prob = 1.
    val_params.decoder_keep_probs = [1., 1., 1.]
    return val_params

def create_graph(feeder, hparams, mode = 'train'):
    '''
    feeder: time serieses feeder.
    hparams: hyper parameters on which dataset pipeline and model params are defined.
    mode: train or test mode.
    
    '''
    if mode == 'validation':
        model_params = switch_to_validation_mode(hparams)
    else:
        model_params = hparams
        
    graph = tf.Graph()
    with graph.as_default():
        pipeline = InputPipeLine(feeder, model_params)
        history_model = model_builder(pipeline, model_params)
    return history_model, graph, pipeline, model_params


class make_checkpoint:
    '''
    saves the model into a checkpoint file if its accuracy is the highest.
    '''
    def __init__(self, hparams, saver_mode = 'min', await_epochs = 10):
        self.await_epochs = await_epochs
        self.saver = tf.train.Saver(max_to_keep = 3)
        self.chechpoint_path = hparams.best_ckpt_path
        self.every_ckpt_path = hparams.every_ckpt_path
        self.accuracies = []
        self.end_epochs = False
        self.saver_mode = saver_mode
        
    def add(self, sess, epoch, accuracy):
        self.accuracy = accuracy
        self.epoch = epoch
        self.accuracies.append(self.accuracy)
        if self.saver_mode == 'max':
            self.best_index = self.accuracies.index(max(self.accuracies))+1
        elif self.saver_mode == 'min':
            self.best_index = self.accuracies.index(min(self.accuracies))+1
        self.current_index = self.epoch
        
        if self.current_index == self.best_index: self.save_model(session = sess, mode = 'best')
        
    def save_model(self, session, mode):
        if mode == 'best':
            self.saver.save(session, self.chechpoint_path)
        
        elif mode == 'every':
            self.saver.save(session, self.every_ckpt_path)
    
    def load_model(self, session):
        self.saver.restore(session, self.every_ckpt_path)
            
def check_paths(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)
                
