# ASHARE-Great-Energy-Predictor-III
An Implementation of Deep Learning Sequence model to predict Energy Consumption using LSTM.

This model uses LSTM to predict the future energy consumption based on the history readings.
Seq-to-Seq structure, with the use of attention network and is written in Tensorflow.
# Features used:
1) month (one hot)
2) Day of week (one hot)
3) meter (one hot)
4) time lags
5) six-month auto correlation co-officient.

the unique features of each timeseries are tiled and concatenated to each time step featureas.
the encoder is CuDNNLSTM which is much faster than regular tensorflow LSTM Layer and the decoder is a LSTM cell wrapped in a tf.while_loop, which attends to the encoder states at each prediction timestep.

# Resampling data:
since the data is provided for a whole year hour by hour, it will be difficult to predict long periods (3 months for example) at once, as each day has 24 time steps which is too long for prediction. to solve this issue, the model provides an option to resamle each day in the timeseries into n hours steps such that if the resampling steps are 4 then the day will be divided into 4 readings instead of 24, where each reading represent a mean of 6 hours. this step can be used to provide an approximate solution to predict longer sequences.
To avoid resampling just put the resampling_steps to 24.

Note: you should always put the time parameters in days, the data generator will account for the resampling process. 

you can just run the training and validation phase by setting the model_mode parameter into 'train' (default) and running the main.py file

TODO: upload the test.py file
