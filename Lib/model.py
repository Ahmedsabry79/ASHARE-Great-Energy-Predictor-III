# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:46:22 2019

@author: A-MSabry
"""

import tensorflow as tf
from tensorflow.python.util import nest
from Lib.losses import RMSLE_denorm_loss, RMSE_denorm_loss, RMAE_denorm_loss, RMSE_loss, RMAE_loss

def build_lstm_encoder(x, units, num_layers, encoder_keep_prob):
    '''
    x: the input sequence [batch_size, time_steps, depth]
    units: LSTM Units
    num_layers: Number of LSTM Layers in the encoder
    encoder_keep_prob: keep_prob of the dropout layer of encoder output
    returns: 
    '''
    outputs = []
    h_states = []
    c_states = []
    
    layer = [tf.keras.layers.CuDNNLSTM(units, return_sequences = True, return_state = True) \
             for i in range(num_layers)]
    
    for i in range(num_layers):
        if i == 0:
            Output, h_state, c_state = layer[i](x)
            outputs.append(tf.nn.dropout(Output, keep_prob = encoder_keep_prob))
            h_states.append(h_state)
            c_states.append(c_state)
        else:
            Output, h_state, c_state = layer[i](outputs[-1])
            outputs.append(Output)
            h_states.append(h_state)
            c_states.append(c_state)
        
        states = tuple([tf.nn.rnn_cell.LSTMStateTuple(c = a, h = b) for a, b in zip(c_states, h_states)])
            
    return outputs[-1], states


def ED_state_converter(states, keep_prob, model_layers, model_mode, decoder_units, 
                       map_encoder_final_states = False):
    """
    converts the encoder final states dimensions into the dimensions matching the decoders'
    states, also wraps dropout.
    """
        
    def wrap_dropout(st, kb):
        new_st = nest.map_structure(lambda x: tf.nn.dropout(x, kb), st)
        return new_st
    
    def apply(States):
        States = wrap_dropout(States, keep_prob)
        
        if map_encoder_final_states == True:
            mapped_states = wrap_dropout(nest.map_structure(lambda x: tf.keras.layers.Dense(decoder_units)(x), States), keep_prob)

            if model_layers == 1:
                return mapped_states[0]
            return tuple(mapped_states)
        
        elif map_encoder_final_states == False:
            if model_layers == 1:
                return States[0]
            return States
        
    new_states = apply(states)
    return new_states
    

def build_lstm_decoder(units, num_layers, encoder_states, encoder_end_states, encoder_units, 
                       encoder_time_steps, prediction_inputs, previous_y, prediction_steps, attn_depth,
                       attn_vec_size, decoder_keep_probs, model_mode):
    '''
    units: number of hidden units in the LSTM Cell
    num_layers: number of LSTM layers of the decoder
    encoder_states: the hidden states of the encoder -> [batch_size, time_steps, depth]\
    (states of top LSTM Layer and used to calculate attention state).
    encoder_end_states: Last c and h states for each LSTM Encoder Layers.
    prediction_inputs: the input features at different time steps of the decoder -> [batch_size, time_steps, depth]
    previous_y: last time series value that the encoder should predict to be fed in the decoder\
                used to stabilize the decoder to train to take the output as input for the next sequence.
    prediction_steps: Prediction time steps.
    attn_depth: depth of the attention state.
    
    '''
    ## Calculates the input size of the LSTM Cell
    input_size = prediction_inputs.get_shape().as_list()[-1] + 1 + attn_depth

    ## Builds the cell
    def build_lstm(units, decoder_keep_probs):
        Cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(units),
                                             dtype=tf.float32, input_size = input_size,
                                             input_keep_prob = decoder_keep_probs[0] , 
                                             output_keep_prob = decoder_keep_probs[1] , 
                                             state_keep_prob = decoder_keep_probs[2] )
        
        return Cell
    
    if num_layers == 1:
        cell = build_lstm(units, decoder_keep_probs)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([build_lstm(units, decoder_keep_probs) for i in range(num_layers)])

    ## Builds the attention:
    def Attention(query, states, time_steps, attn_vec_size):
        '''
        query: h state of the previous step.
        states: encoder states.
        time_steps: encoder time steps.
        attn_vec_size:
        '''
        conv2d = tf.nn.conv2d
        reduce_sum = tf.reduce_sum
        softmax = tf.nn.sigmoid
        tanh = tf.nn.tanh
        Linear = tf.keras.layers.Dense(attn_vec_size)
        
        with tf.variable_scope("attention"):
            k = tf.get_variable('k', [1, 1, encoder_units, attn_vec_size])
            v = tf.get_variable('v', [attn_vec_size])
        
        hidden = tf.reshape(states,
                            [-1, time_steps, 1, encoder_units])

        hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
          
        y = Linear(query)
        y = tf.reshape(y, [-1, 1, 1, attn_vec_size])
        s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
        a = softmax(s)
        d = reduce_sum(
            tf.reshape(a, [-1, time_steps, 1, 1]) * hidden, [1, 2])
        new_attns = tf.reshape(d, [-1, attn_depth])
        
        return new_attns
    
    def map_decoder_output(cell_output):
        return tf.keras.layers.Dense(1)(cell_output)
    
    ## Defining the loop condition function:
    def cond_fn(time, *args):
        return time < prediction_steps
    
    def loop_fn(time, prev_output, prev_state, decoder_states, decoder_outputs):
        '''
        time: the decoder time step
        prev_output: the prediction value "yt" at the previous time step
        prev_state: the hidden state from the previous time step (previous cell state)
        array_targets: the tensor array used to record targets (decoder states)
        array_outputs: the tensor array used to record outputs (decoder outputs)
        '''

        def get_prev_h_state(states):
            if num_layers == 1:
                return states[1]
            return states[0][1]
        
        prev_h_state = get_prev_h_state(prev_state)

        attention_state = Attention(prev_h_state, encoder_states, encoder_time_steps, attn_vec_size)

        ## Prepare the input for this step:
        cell_input = tf.concat([prediction_inputs[:, time, :], 
                                prev_output, 
                                attention_state], axis = 1)

        new_out, new_state = cell(cell_input, prev_state)

        ## then write the state to the state array 
        decoder_states = decoder_states.write(time, new_state)

        ## then map the state to get the output and write it to the output array
        final_output = map_decoder_output(new_out)

        decoder_outputs = decoder_outputs.write(time, final_output)

        return time+1, final_output, new_state, decoder_states, decoder_outputs

    init_conditions = [tf.constant(0, dtype = tf.int32),
                       previous_y, 
                       encoder_end_states,
                       tf.TensorArray(tf.float32, size = prediction_steps),
                       tf.TensorArray(tf.float32, size = prediction_steps)]
    
    _, _, _, decoded_states, decoded_outputs = tf.while_loop(cond_fn, loop_fn, init_conditions)
    decoded_states, decoded_outputs = decoded_states.stack(), decoded_outputs.stack()
    
    ## [n_layers, BS, c/h, T, D] or [BS, c/h, T, D] 
    decoded_states = tf.transpose(decoded_states, [2, 1, 0, 3]) if num_layers == 1 \
                     else tf.transpose(decoded_states, [1, 3, 2, 0, 4]) 
    ## [BS, T]
    decoded_outputs = tf.squeeze(tf.transpose(decoded_outputs, [1, 0, 2]), axis = 2) 
    
    return decoded_outputs, decoded_states
    
    
class model_builder:
    
    def __init__(self, inp, hparams):
    
        self.inp = inp
        self.hparams = hparams
        
        self.hparams.map_encoder_final_states = True if self.hparams.encoder_units != self.hparams.decoder_units \
                                                     else self.hparams.map_encoder_final_states
                                                     
        self.optimization_dict = {'adam': tf.train.AdamOptimizer,
                                  'adagrad': tf.train.AdagradOptimizer,
                                  'rmsprob': tf.train.RMSPropOptimizer,
                                  'gradient_descent': tf.train.GradientDescentOptimizer}
        
        self.losses_dict = {'mse': RMSE_loss,
                            'mae': RMAE_loss}
        
        self.metrics_dict = {"dmsle": RMSLE_denorm_loss,
                             "dmse": RMSE_denorm_loss,
                             "dmae": RMAE_denorm_loss}
        
        self.global_step = tf.Variable(0, trainable = False, name = 'global_step')
        
        self._build_graph()
        if self.inp.model_mode != 'test':
            self._compute_loss()
        if self.inp.model_mode == 'train':
            self._build_optimizer()
        if self.inp.model_mode != 'test':
            self._compute_metrics()
        
        self.saver = tf.train.Saver(max_to_keep = 1)
        
    def _build_graph(self):
        self.encoder_outputs, self.encoder_states = build_lstm_encoder(self.inp.x_features, self.hparams.encoder_units, 
                                                                       self.hparams.model_layers, self.hparams.encoder_keep_prob, 
                                                                       )
        
        self.encoder_states = ED_state_converter(self.encoder_states, self.hparams.enc_dec_keep_prob, 
                                                 self.hparams.model_layers, self.inp.model_mode, 
                                                 self.hparams.decoder_units, self.hparams.map_encoder_final_states)
        
        self.decoder_outputs, self.decoder_states = build_lstm_decoder(self.hparams.decoder_units, self.hparams.model_layers, 
                                                                       self.encoder_outputs, self.encoder_states, 
                                                                       self.hparams.encoder_units, self.inp.encoder_steps, 
                                                                       self.inp.y_features, 
                                                                       tf.reshape(self.inp.x_hits[:, -1], [-1, 1]), 
                                                                       self.inp.decoder_steps, self.hparams.encoder_units,
                                                                       self.hparams.attn_vec_size, self.hparams.decoder_keep_probs, 
                                                                       self.inp.model_mode)
    
    def _build_optimizer(self):
        if self.hparams.exp_lr_decay:
            lr = tf.train.exponential_decay(self.hparams.learning_rate,
                                            self.global_step,
                                            self.hparams.decay_steps,
                                            decay_rate = 0.995,
                                            staircase=True)
        else: lr = self.hparams.learning_rate
        self.optimizer = self.optimization_dict[self.hparams.optimizer](lr)
        
        ## Gradients computation:
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        if self.hparams.clip_norm != None:
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.hparams.clip_norm)
            self.update_op = self.optimizer.apply_gradients(zip(clipped_grads, trainable_variables),
                                                            global_step = self.global_step)
        else:
            self.update_op = self.optimizer.apply_gradients(zip(grads, trainable_variables),
                                                            global_step = self.global_step)
            
    def _compute_loss(self):
        
        loss = self.losses_dict[self.hparams.loss](y_true = self.inp.y_out, 
                                                   y_pred = self.decoder_outputs)
        self.loss = loss
    
    
    def _compute_metrics(self):
        if self.hparams.metrics != None:
            metrics = self.hparams.metrics
            metrics_values = [self.metrics_dict[i](self.inp.y_hits, self.decoder_outputs, 
                                                   self.inp.means, self.inp.stds, 
                                                   self.hparams.epsilon) for i in metrics]
            self.metrics =  dict(((i, j) for i, j in zip(metrics, metrics_values)))
        else:
            self.metrics = None
            
    