# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:59:54 2019

@author: A-MSabry
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

class DataPreparator:
    
    def load_df(self):
        '''
        loads the train file, resamples it and groups it into time serieses.
        '''
        train = pd.read_csv(self.train_url, low_memory = False)
        meta = pd.read_csv(self.metadata_url)
        train = pd.merge(train, meta, how = 'left', on = 'building_id').drop(
                ['square_feet', 'year_built', 'floor_count', 'site_id'], axis = 1)
        train.timestamp = pd.to_datetime(train.timestamp)
        train = train.set_index('timestamp')
        self.time = train.resample(str(24//self.resampling_steps)+"H").mean().sort_index().index
        time_serieses = train.groupby(['building_id', 'meter'])
        self.groups = list(time_serieses.groups.keys())
        self.extract_time_variables()
        
        return time_serieses
    
    def extract_time_variables(self):
        '''
        dow: day of week
        '''
        self.dows = np.array([self.dow_dict[i.weekday()] for i in self.time])
        self.months = np.array([self.month_dict[i.month] for i in self.time])
        self.time_features = np.concatenate([self.months, self.dows], axis = 1)
        
    def extract_variables(self, time_serieses, group):
        '''
        this function is applied to each single time series, it calculates the 6-month
        auto_corr of the time series, gets the unique building features, gets lags and 
        clips the timeseries to get rid of outliers.
        '''
        df = time_serieses.get_group(group).sort_index()
        puse = self.PUse_dict[df.primary_use[0]]
        meter = self.meter_dict[df.meter[0]]
        
        hits = df.resample(str(24//self.resampling_steps)+"H").mean()['meter_reading']\
        .fillna(method = 'ffill') if self.resampling_steps != 24 else \
        df.resample(str(24//self.resampling_steps)+"H").mean()['meter_reading']\
        .fillna(value = 0)
        
        if self.outliers_thresh != None:
            hits = self.correct_outliers(hits, self.outliers_thresh)
        
        time_df = pd.DataFrame({"Date": self.time})
        time_df = time_df.set_index("Date")
        padded_hits = time_df.join(hits).fillna(0)
        
        autocorr = self.get_autocorr(hits, 6*30*self.resampling_steps)
        shift3m = padded_hits.shift(30*3*self.resampling_steps).fillna(0)
        shift6m = padded_hits.shift(30*6*self.resampling_steps).fillna(0)
        return padded_hits, shift3m, shift6m, autocorr, meter, puse
    
    def make_features(self):

        self.hits_set = []
        self.building_features = []
        self.lags_set = []
        self.groups_set = []
        time_serieses = self.load_df()
        
        for group in self.groups:
            hits, shift3m, shift6m, autocorr, meter, puse = self.extract_variables(time_serieses, group)
            lags = np.stack((shift3m.values.reshape(-1), shift6m.values.reshape(-1)), 0)
            self.hits_set.append(hits.values.reshape(-1))
            self.building_features.append(meter + puse + [autocorr])
            self.lags_set.append(lags)
            self.groups_set.append(group)
            
        self.hits_set = np.array(self.hits_set)
        self.building_features = np.array(self.building_features)
        self.lags_set = np.array(self.lags_set)
        self.groups_set = np.array(self.groups_set)
        
    def get_autocorr(self, series, lag):
        """
        Autocorrelation for single data series
        :param series: time series
        :param lag: lag, days
        :return:
        """
        s1 = series[lag:]
        s2 = series[:-lag]
        ms1 = np.mean(s1)
        ms2 = np.mean(s2)
        ds1 = s1 - ms1
        ds2 = s2 - ms2
        divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
        return np.sum(ds1 * ds2) / divider if divider != 0 else 0
    
    def correct_outliers(self, hits, thresh):
        
        ## Dummy calculation of non-zero mean:
        dummy_hit_for_mean = hits.copy()
        dummy_mean = dummy_hit_for_mean[dummy_hit_for_mean > 0].mean()
        
        ## Fill the nan values with zeros can affect z score and outliers calculation
        ## while mean will not affect them
        dummy_hits = hits.copy().fillna(value = dummy_mean).values.reshape(-1)
        dummy_hits[dummy_hits == 0] = dummy_mean
        dummy_z = zscore(dummy_hits)
    
        idx = np.where((dummy_z < thresh) & (dummy_z > -thresh))[0]
        max_value = dummy_hits[idx].max()
        min_value = dummy_hits[idx].min()
        
        outlier_pos = np.where(dummy_z > thresh)[0]
        outlier_neg = np.where(dummy_z < -thresh)[0]
        
        ## to keep the zeros (missing) values without changes.
        zeros = np.where(hits == 0)[0]
    
        corrected_hits = hits.copy().fillna(0).values
        corrected_hits[outlier_pos] = max_value
        corrected_hits[outlier_neg] = min_value
        corrected_hits[zeros] = 0.
    
        return pd.Series(corrected_hits, index = hits.index, name = 'meter_reading')
    
    
    def __init__(self, hparams):
        self.train_url = hparams.train_url
        self.metadata_url = hparams.metadata_url
        self.resampling_steps = hparams.resampling_steps
        self.outliers_thresh = hparams.outliers_thresh
        self.define_categoricals()
        self.make_features()
        
        
    def define_categoricals(self):
        self.PUse_dict = {'Other': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Education': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Lodging/residential': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Office': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Entertainment/public assembly': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Retail': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Parking': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Public services': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          'Warehouse/storage': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                          'Food sales and service': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          'Religious worship': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                          'Healthcare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                          'Utility': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                          'Technology/science': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          'Manufacturing/industrial': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                          'Services': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                           }
        
        self.meter_dict = {0: [0, 0, 0],
                           1: [1, 0, 0],
                           2: [0, 1, 0],
                           3: [0, 0, 1]}
        
        self.dow_dict = {0: [0, 0, 0, 0, 0, 0],
                         1: [1, 0, 0, 0, 0, 0],
                         2: [0, 1, 0, 0, 0, 0],
                         3: [0, 0, 1, 0, 0, 0],
                         4: [0, 0, 0, 1, 0, 0],
                         5: [0, 0, 0, 0, 1, 0],
                         6: [0, 0, 0, 0, 0, 1]}
        
        self.month_dict = {1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           2: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           4: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           5: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           6: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           7: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           8: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           9: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}



