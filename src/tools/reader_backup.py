# -*- coding: utf-8 -*-

from sparnn.iterators import DataIterator

import numpy as np
from utils import *

import theano


class WeatherIterator(DataIterator):
    def __init__(self, iterator_param,  mode='train'):
        self.use_input_mask = iterator_param.get('use_input_mask', None)
        self.use_output_mask = iterator_param.get('use_output_mask', None)
        self.name = iterator_param['name']
        self.input_data_type = iterator_param.get('input_data_type', theano.config.floatX)
        self.output_data_type = iterator_param.get('output_data_type', theano.config.floatX)
        self.minibatch_size = iterator_param['minibatch_size']
        self.is_output_sequence = iterator_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0

        self.config = iterator_param
        self.mode = mode

        self.load()

    def load(self):
        self.data = self.raw_input(self.config)
        self.check_data()

    def raw_input(self, config):
        filenames = load_range_filenames(config)
        print("image length: ", len(filenames))

        X, X_last = [], []
        for i, filename in enumerate(filenames): # filenames = [201811230524.png, ..., ]
            #print(filename)
            X_seq = filenames[i: i+config['input_seq_length']+config['output_seq_length']]
            if len(X_seq) == (config['input_seq_length']+config['output_seq_length']):
                print("X_seq: ", X_seq)
                for filename1 in X_seq:
                    try:
                        X_hour = read_data(filename1, config) # X_hour is compressed point matrix
                        #print(X_hour)
                        if X_hour is not None:
                            X.append(X_hour)
                            X_last = X_hour
                        elif len(X_last) > 0:
                            X.append(X_last)
                    except IOError:
                        print(filename+' not exists!')
                        pass
            else:
                break
            #if i % 240 == 0 and i > 1:  # change day  original:i % 240
                 #print('read to ', i, filename, len(filenames))
        
        X = np.array(X, dtype=np.dtype("float32"))
        # print('read X', X.shape)
        X = np.reshape(X, (X.shape[0], 1, X[0].shape[0], X[0].shape[1])) # ???

        #print('load ', X.shape, X.max(), X.min())

        # normalize
        X = (X+config['offset'])/config['max']  # todo:Normalization

        #print('normalize ', X.shape, X.max(), X.min())


        clips = [[] for i in range(2)]  # [[],[]]
        minibatch_size = config['minibatch_size']
        input_seq_length = config['input_seq_length']
        output_seq_length = config['output_seq_length']
        # print('minibatch_size:', minibatch_size)
        # print('input_seq_length:', input_seq_length)
        # print('output_seq_length:', output_seq_length)

        
        if self.mode == 'train':  # load input+output
            #print(X.shape)
            #print('input+output:',input_seq_length+output_seq_length)
            for x in range(X.shape[0]/(input_seq_length+output_seq_length)):
                #print(x)
                #print([(input_seq_length+output_seq_length)*x, input_seq_length])
                clips[0].append([(input_seq_length+output_seq_length)*x, input_seq_length])
                #print([(input_seq_length+output_seq_length)*x+input_seq_length, output_seq_length])
                clips[1].append([(input_seq_length+output_seq_length)*x+input_seq_length, output_seq_length])
        elif self.mode == 'predict': # load input only
            for x in range(X.shape[0]/(input_seq_length)):
                # print(x)
                clips[0].append([(input_seq_length+output_seq_length)*x, input_seq_length])
                clips[1].append([(input_seq_length+output_seq_length)*x+input_seq_length, 0])
        clips = np.array(clips, dtype=np.dtype("int32"))
        #print('clips:', clips)
        dims = np.array([[1, X[0].shape[1], X[0].shape[2]]], dtype=np.dtype("int32"))
        #print('dims:', dims)
        return {'input_raw_data': X, 'dims': dims, "clips": clips}

def load_data(config, mode='predict'):

    iterator = WeatherIterator(config, mode=mode)
    iterator.begin(do_shuffle=True)

    return iterator
