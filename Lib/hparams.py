# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 01:33:40 2019

@author: A-MSabry
"""
import argparse


def model_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_layers", type = int, default = 1)
    parser.add_argument("--encoder_units", type = int, default = 256)
    parser.add_argument("--decoder_units", type = int, default = 256)
    parser.add_argument("--attn_vec_size", type = int, default = 64)
    parser.add_argument("--encoder_keep_prob", type = float, default = 0.8)
    parser.add_argument("--enc_dec_keep_prob", type = float, default = 0.8)
    parser.add_argument("--decoder_keep_probs", type = list, default = [0.8, 0.8, 0.8])
    parser.add_argument("--map_encoder_final_states", type = bool, default = False)
    parser.add_argument("--loss", type = str, default = 'mse')
    parser.add_argument("--optimizer", type = str, default = 'adam')
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--exp_lr_decay", type = bool, default = False)
    parser.add_argument("--decay_steps", type = int, default = 1000)
    parser.add_argument("--clip_norm", type = float, default = None)
    parser.add_argument("--epsilon", type = float, default = 1e-8)
    parser.add_argument("--metrics", type = float or list, default = None)
    parser.add_argument("--total_period", type = int, default = 366)
    parser.add_argument("--training_period", type = int, default = 300)
    parser.add_argument("--train_window", type = int, default = 30)
    parser.add_argument("--pred_window", type = int, default = 7)
    parser.add_argument("--batching_interval", type = int, default = 4)
    parser.add_argument("--free_random_space", type = int, default = 15)
    parser.add_argument("--resampling_steps", type = int, default = 24)
    parser.add_argument("--model_mode", type = str, default = 'train')
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--Zeros_Thresh", type = float, default = 0.05)
    parser.add_argument("--std_thresh", type = float, default = 3.5)
    parser.add_argument("--missing_mean", type = float, default = 500)
    parser.add_argument("--missing_std", type = float, default = 200)
    parser.add_argument("--outliers_thresh", type = float, default = 4.)
    parser.add_argument("--train_url", type = str, default = "./Data/train.csv")
    parser.add_argument("--metadata_url", type = str, default = "./Data/building_metadata.csv")
    parser.add_argument("--best_ckpt_path", type = str, default = "./ckpts/best/")
    parser.add_argument("--every_ckpt_path", type = str, default = "./ckpts/every/")
    parser.add_argument("--logs_dir", type = str, default = "./logs/")
    
    para = parser.parse_args()
    return para

