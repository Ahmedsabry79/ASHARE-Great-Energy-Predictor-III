# ASHARE-Great-Energy-Predictor-III
An Implementation of Deep Learning Sequence model to predict Energy Consumption using LSTM.

this model uses LSTM to predict the future energy consumption based on the history readings.
this model uses Seq-to-Seq structure, with the use of attention network.
# the features used are:
1) month (one hot)
2) Day of week (one hot)
3) meter (one hot)
4) time lags
5) auto correlation co-officient.

the unique features of each timeseries are tiled and concatenated to the timeseries featureas.
the encoder is CuDNNLSTM which is much faster than regular tensorflow LSTM Layer and the decoder is a LSTM cell wrapped in a tf.while_loop, which attends to the encoder states at each prediction timestep.

# Resampling data:
since the data is provided for a whole year hour by hour, it will be difficult to predict it all at once, so the model provides an option to resamle the timeseries into n hours steps such that if the resamplin steps are 4 then the day will be divided into 4 readings instead of 24, where each reading represent a mean of 6 hours. this step can be used to provide an approximate solution.

you can just run the training and validation phase by setting the model_mode parameter into 'train' (default) and running the main.py file
TODO: upload the test.py file
