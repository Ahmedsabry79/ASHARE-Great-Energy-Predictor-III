# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:29:34 2019

@author: A-MSabry
"""

import tensorflow as tf
from Lib.utils import create_graph, make_checkpoint, set_logger
import os

def train(data_feeder, train_model, train_sess, train_graph, train_pipeline, train_params):
    
    val_model, val_graph, val_pipeline, val_params = create_graph(data_feeder, 
                                                                  train_params, 
                                                                  mode = 'validation')
    
    with tf.Session(graph = val_graph) as val_sess:
        
        checkpoint_maker = make_checkpoint(train_params, await_epochs = 3)
        print('training has started')
        summary_logger = set_logger(logger_name = os.path.join(train_params.logs_dir, "summary_logger"), level = 'info')
        warning_logger = set_logger(logger_name = os.path.join(train_params.logs_dir, "warning_logger"), level = 'warning')
        
        for epoch in range(1, train_params.epochs+1):
            epoch_logger = set_logger(os.path.join(train_params.logs_dir, +"epoch {}".format(epoch)), level = 'info')
            
            print("epoch {} training phase has started.".format(epoch))
            train_sess.run(train_pipeline.iter.initializer)
            epoch_loss = 0
            counter = 0
            if train_model.metrics != None:
                metrics_names = list(train_model.metrics.keys())
                train_metrics_values = list(train_model.metrics.values())
                epoch_train_metrics = dict(((i, 0) for i in metrics_names))

            while True:
                statements = []
                if train_model.metrics != None:
                    try:
                        _, c, step, training_metrics = train_sess.run([train_model.update_op, 
                                                                       train_model.loss, 
                                                                       train_model.global_step, 
                                                                       train_metrics_values])
                        
                        epoch_loss += c
                        for k in range(len(train_metrics_values)):
                            epoch_train_metrics[metrics_names[k]] += training_metrics[k] 
                        counter += 1
                        statements.append('epoch {}, Batch {}, batch loss: {}'.format(epoch, counter, round(c, 4)))

                        for i in range(len(train_metrics_values)):
                            statements.append('{}: {}'.format(metrics_names[i], round(training_metrics[i], 4)))
                        epoch_logger.info(' '.join(statements))
                        print("Batch {}".format(counter), end = '')
                        print("\r", end = '')
                    except tf.errors.OutOfRangeError:
                        train_loss = epoch_loss / counter
                        for k in range(len(train_metrics_values)):
                            epoch_train_metrics[metrics_names[k]] /= counter 
                        train_model.saver.save(train_sess, train_params.every_ckpt_path)
                        
                        break
                else:
                    try:
                        print("epoch {} training phase has started.".format(epoch))
                        _, c, step = train_sess.run([train_model.update_op, 
                                                     train_model.loss, 
                                                     train_model.global_step])
                        
                        epoch_loss += c
                        counter += 1
                        epoch_logger.info('epoch {}, Batch {}, batch loss: {}'.format(epoch, counter, round(c, 4)))
                        print("Batch {}".format(counter))
                        print("\r", end = ' ')
                    except tf.errors.OutOfRangeError:
                        train_loss = epoch_loss / counter
                        train_model.saver.save(train_sess, train_params.every_ckpt_path)
                        
                        break
            
            ## Validation:
            checkpoint_maker.load_model(val_sess)
            val_sess.run(val_pipeline.iter.initializer)
            
            val_epoch_loss = 0
            val_counter = 0
            val_metrics_values = list(val_model.metrics.values())
            epoch_val_metrics = dict(((i, 0) for i in metrics_names))
            
            print("epoch {} Validation phase has started.".format(epoch))
            while True:
                if train_model.metrics != None:
                    try:
                        c, validation_metrics = val_sess.run([val_model.loss, val_metrics_values])
                        if c > 3:
                            warning_logger.warn("Some std are very low making bad loss calculations.")
                        val_epoch_loss += c
                        val_counter += 1
                        for k in range(len(val_metrics_values)):
                                epoch_val_metrics[metrics_names[k]] += validation_metrics[k] 
    
                    except tf.errors.OutOfRangeError:
                        val_loss = val_epoch_loss/val_counter
                        for k in range(len(val_metrics_values)):
                            epoch_val_metrics[metrics_names[k]] /= val_counter 
                        
                        checkpoint_maker.add(val_sess, epoch = epoch, accuracy = val_loss)
                        summary_logger.info('epoch {}, epoch train loss: {}, epoch val loss {}'\
                                            .format(epoch, round(train_loss, 4), val_loss))
                        summary_logger.info("training metrics: ")
                        
                        for i in epoch_train_metrics.keys():
                            summary_logger.info("{}: {}".format(i, round(epoch_train_metrics[i], 4)))
                        
                        summary_logger.info("validation metrics: ")

                        for i in epoch_val_metrics.keys():
                            summary_logger.info("{}: {}".format(i, round(epoch_val_metrics[i], 4)))
                        break
                else:
                    print("epoch {} training phase has started.".format(epoch))
                    try:    
                        c = val_sess.run([val_model.loss])
                        if c[0] > 3:
                            warning_logger.warn("Some std are very low making bad loss calculations.")
                            
                        val_epoch_loss += c
                        val_counter += 1
                        
                    except tf.errors.OutOfRangeError:
                        val_loss = val_epoch_loss/val_counter
                        checkpoint_maker.add(val_sess, epoch = epoch, accuracy = val_loss)
                        summary_logger.info('epoch {}, epoch train loss: {}, epoch val loss {}'\
                                            .format(epoch, round(train_loss, 4), val_loss))
                                                
                        break





