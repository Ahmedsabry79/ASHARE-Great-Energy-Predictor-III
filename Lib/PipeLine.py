# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:00:05 2019

@author: A-MSabry
"""

import tensorflow as tf

train_url = r'C:/Users/User/Desktop/LSTM Model/Data/train.csv'
metadata_url = r'C:/Users/User/Desktop/LSTM Model/Data/building_metadata.csv'


class InputPipeLine:
    
    def splitter(self, hits, lags):
        '''
        splits the whole timeseries to train and val segments applying
        walk-forward validation strategy.
        '''
        train_hits, val_hits = hits[:self.training_period], hits[-self.val_period:]
        train_lags, val_lags = lags[:, :self.training_period], lags[:, -self.val_period:]
        
        train_time_features, val_time_features = self.feeder.time_features[:self.training_period], \
                                                 self.feeder.time_features[-self.val_period:]
        
        return train_hits, train_lags, train_time_features, val_hits, val_lags, val_time_features
    
    
    def windower(self, true_hits, true_lags, time_features, building_features, 
                 true_period, mean, std):
            '''
            splits each timeseries into sub-timeserieses.
            '''
            x_samples = []
            y_samples = []
            y_out_samples = []
            
            x_hits_samples = []
            y_hits_samples = []
            
            x_means = []
            x_stds = []

            length = true_period - self.train_window - self.pred_window
            
            if self.hparams.epsilon != None:
                std = std + self.hparams.epsilon
            else:
                std = std
            hits_normed = (true_hits - mean)/std
            lags_normed = (true_lags - mean)/std 

            features = tf.concat([tf.expand_dims(hits_normed, 1), 
                                  tf.transpose(lags_normed, [1, 0]), 
                                  tf.cast(time_features, tf.float64), 
                                  tf.cast(tf.tile(tf.expand_dims(building_features, 0), 
                                                  [true_period , 1]), tf.float64),                
                                  ], axis = 1)
                           
            for i in range(0, length, self.batching_interval):
                
                start = i
                x_end = start + self.train_window
                y_end = start + self.train_window + self.pred_window
                
                x_sample = features[start: x_end]
                y_sample = features[x_end: y_end][:, 1:]
                y_out = features[x_end: y_end][:, 0]
                x_hits = true_hits[start: x_end]
                y_hits = true_hits[x_end: y_end]
                
                
                x_samples.append(x_sample)
                y_samples.append(y_sample)
                y_out_samples.append(y_out)
                x_hits_samples.append(x_hits)
                y_hits_samples.append(y_hits)
                x_means.append(mean)
                x_stds.append(std)

            return x_samples, y_samples, y_out_samples, x_hits_samples, y_hits_samples, x_means, x_stds
        
        
    def reject_filter(self, x_hits, y_hits, x_features, y_features, y_out, means, stds):
        
        count = tf.reduce_sum(tf.cast(tf.equal(x_hits, 0.), tf.int32))
        keep = count <= tf.cast(self.Zeros_Thresh * self.train_window, tf.int32)

        return keep
    
    def reject_std_filter(self, x_hits, y_hits, x_features, y_features, y_out, means, stds):
        keep = stds >= self.std_thresh
        return keep
    
    def make_train_features(self, hits, lags, building_features):        
        '''
        combines hits, time lags and building features into a long timeseries
        and then cuts the long timeseries into smaller batches using windower func.
        the long timeseries is given a random starting point for each timeseries
        to provide some sort of data augmentation.
        '''
        true_period = self.training_period - self.free_random_space
        start_index = tf.cast(tf.random_uniform(shape = (), minval=0, maxval=self.free_random_space), tf.int32)

        end_index = start_index + true_period
        
        hits, lags, time_features, _, _, _ = self.splitter(hits, lags)
        true_indices = tf.not_equal(hits, 0)
        mean_std_hits = tf.boolean_mask(hits, true_indices)
        
        def true_fn():
            hits_mean, hits_var = tf.nn.moments(tf.expand_dims(mean_std_hits, 1),axes = [0, 1])
            hits_std = tf.sqrt(tf.cast(hits_var, tf.float64))
            hits_mean = tf.cast(hits_mean, tf.float64)
            return hits_mean, hits_std
        
        def false_fn():
            hits_std = self.hparams.missing_mean
            hits_mean = self.hparams.missing_std
            return tf.cast(hits_mean, tf.float64), tf.cast(hits_std, tf.float64)
        
        hits_mean, hits_std = tf.cond(tf.reduce_sum(mean_std_hits) > 0, true_fn, false_fn)
        
        true_hits = hits[start_index: end_index]
        
        true_lags = lags[:, start_index: end_index]

        time_features = tf.constant(time_features)[start_index: end_index]

        x_features, y_features, y_out, x_hits, y_hits, means, stds = self.windower(true_hits, true_lags, 
                                                                                   time_features, 
                                                                                   building_features, 
                                                                                   true_period, 
                                                                                   hits_mean,
                                                                                   hits_std)
        return x_hits, y_hits, x_features, y_features, y_out, means, stds
    
    def make_val_features(self, hits, lags, building_features):        
        '''
        combines hits, time lags and building features into a long timeseries
        and then cuts the long timeseries into smaller batches using windower func.
        the long timeseries is given a random starting point for each timeseries
        to provide some sort of data augmentation.
        '''
        true_period = self.val_period 
        hits, lags, time_features, val_hits, val_lags, val_time_features = self.splitter(hits, lags)
        true_indices = tf.not_equal(hits, 0)
        mean_std_hits = tf.boolean_mask(hits, true_indices)
        
        def true_fn():
            hits_mean, hits_var = tf.nn.moments(tf.expand_dims(mean_std_hits, 1),axes = [0, 1])
            hits_std = tf.sqrt(tf.cast(hits_var, tf.float64))
            hits_mean = tf.cast(hits_mean, tf.float64)
            return hits_mean, hits_std
        
        def false_fn():
            hits_std = self.hparams.missing_mean
            hits_mean = self.hparams.missing_std
            return tf.cast(hits_mean, tf.float64), tf.cast(hits_std, tf.float64)
        
        hits_mean, hits_std = tf.cond(tf.reduce_sum(mean_std_hits) > 0, true_fn, false_fn)
        
        x_features, y_features, y_out, x_hits, y_hits, means, stds = self.windower(val_hits, val_lags, 
                                                                                   val_time_features, 
                                                                                   building_features, 
                                                                                   true_period, 
                                                                                   hits_mean,
                                                                                   hits_std)
        return x_hits, y_hits, x_features, y_features, y_out, means, stds
    
    
    def __init__(self, feeder, hparams):
        
        '''
        Note: any time related argument is put in days and the model takes care of modifying it
              to its actual value.
        This class is responsible for automatic batching and feeding the data into the model.
        args:
        feeder: the variables feeder to the pipeline.
        hparams: hyper parameters object.
        train_window: training Period. 
        pred_window: prediction Period.
        free_random_space: the free space interval to start the division of the time series.
        training_period: the part of the timeseries that will be used for training.
        val_period: the part of the timeseries that will be used for validation.
        total_training_period: total length of the timeseries.
        batching_interval: the interval in days by which the training window dividing the long
                           timeseries into smaller ones will move.
        free_random_space: the period in days allowed for random starting point.
        encoder_steps: encoder time steps.
        decoder_steps: decoder time steps.
        Zeros_Thresh: the maximum allowable zeros percent in each timeseries.
        model_mode: train or test modes.
        '''
        self.feeder = feeder
        self.hparams = hparams
        self.total_period = int(hparams.total_period * feeder.resampling_steps)
        self.train_window = int(hparams.train_window * feeder.resampling_steps)
        self.pred_window = int(hparams.pred_window * feeder.resampling_steps)
        self.training_period = int(hparams.training_period * feeder.resampling_steps)
        self.val_period = self.total_period - self.training_period + self.train_window
        self.batching_interval = int(self.hparams.batching_interval* feeder.resampling_steps)
        self.free_random_space = int(hparams.free_random_space * feeder.resampling_steps)
        self.encoder_steps = self.train_window
        self.decoder_steps = self.pred_window
        self.Zeros_Thresh = float(hparams.Zeros_Thresh)
        self.std_thresh = float(hparams.std_thresh)
        self.batch_size = hparams.batch_size
        self.model_mode = hparams.model_mode
        
        mapper_dict = {'train': self.make_train_features,
                       'validation': self.make_val_features}

        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(feeder.hits_set), 
                                                      tf.constant(feeder.lags_set),
                                                      tf.constant(feeder.building_features)))

        data = dataset.flat_map(lambda x, y, z: tf.data.Dataset\
                               .from_tensor_slices(mapper_dict[self.model_mode](x, y, z))\
                               .filter(self.reject_filter))\
                               .filter(self.reject_std_filter)\
                               .shuffle(self.train_window*feeder.resampling_steps*2000)\
                               .batch(self.batch_size)\
                               .prefetch(10)
        
        iterator = data.make_initializable_iterator()
        batch = iterator.get_next()
        
        self.iter = iterator
        self.x_hits, self.y_hits, self.x_features, self.y_features, self.y_out, self.means, self.stds = batch
        
        self.x_hits = tf.cast(self.x_hits, tf.float32)
        self.y_hits = tf.cast(self.y_hits, tf.float32)
        self.x_features = tf.cast(self.x_features, tf.float32)
        self.y_features = tf.cast(self.y_features, tf.float32)
        self.y_out = tf.cast(self.y_out, tf.float32)
        self.means = tf.cast(self.means, tf.float32)
        self.stds = tf.cast(self.stds, tf.float32)
    
    
        
        
        
        
        
        
        
        
        
        
        
        