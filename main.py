# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 05:27:52 2019

@author: A-MSabry
"""

import tensorflow as tf
from Lib.trainer import train
from Lib.utils import create_graph, check_paths
from Lib.hparams import model_setup
from Lib.Data_Preparation import DataPreparator


tf.reset_default_graph()
model_params = model_setup()
check_paths(model_params.logs_dir, model_params.every_ckpt_path, model_params.best_ckpt_path)
feeder = DataPreparator(model_params)
history_model, train_graph, train_pipeline, train_params = create_graph(feeder, model_params, \
                                                                       mode = model_params.model_mode)

def main():
    with tf.Session(graph = train_graph) as train_sess:
        train_sess.run(tf.global_variables_initializer())
        train(feeder, history_model, train_sess, train_graph, train_pipeline, train_params)

if __name__ == '__main__':
    main()

