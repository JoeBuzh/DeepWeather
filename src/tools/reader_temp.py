# -*- coding: utf-8 -*-
# Modify:Joe-BU

import sys
sys.path.append("/data/buzehao/weather_output20/src/")
from sparnn.iterators import DataIterator
import numpy as np
from utils import *
from mylog import mylog as lg
import theano
from settings import *


class WeatherIterator(DataIterator):
    def __init__(self, iterator_param, mode='train'):
        self.use_input_mask = iterator_param.get('use_input_mask', None)
        self.use_output_mask = iterator_param.get('use_output_mask', None)
        self.name = iterator_param['name']
        self.input_data_type = iterator_param.get(
            'input_data_type', theano.config.floatX)
        self.output_data_type = iterator_param.get(
            'output_data_type', theano.config.floatX)
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
        filenames = load_range_filenames(config, self.mode)
        
        X, X_last = [], []
        if self.mode == 'train':
            load_lg = lg.init_logger(load_log_path)
            load_lg.info(' <<< Start Image : ' + str(filenames[0].split("/")[-1]))
            load_lg.info(' >>> E n d Image : ' + str(filenames[-1].split("/")[-1]))
            load_lg.info(' ** Image Length : ' + str(len(filenames)))
            for i, filename in enumerate(filenames):
                X_seq = filenames[i: i + config['input_seq_length'] + config['output_seq_length']]
                if len(X_seq) == (config['input_seq_length'] + config['output_seq_length']):
                    for filename1 in X_seq:
                        try:
                            # X_hour is compressed point matrix
                            X_hour = read_data(filename1, config)
                            if X_hour is not None:
                                X.append(X_hour)
                                X_last = X_hour
                            elif len(X_last) > 0:
                                X.append(X_last)
                        except IOError:
                            print(filename + ' not exists!')
                            pass
                    load_lg.info('Load to [....' + X_seq[-1].split('/')[-1] + '] done!')
                else:
                    break
            load_lg.info(' Load to X list Done!')
        elif self.mode == 'predict':
            print filenames[0]
            print filenames[-1]
            for i, filename in enumerate(filenames):
                try:
                    # X_hour is compressed point matrix
                    X_hour = read_data(filename, config)
                    if X_hour is not None:
                        X.append(X_hour)
                        X_last = X_hour
                    elif len(X_last) > 0:
                        X.append(X_last)
                except IOError:
                    print(filename + ' not exists!')
                    pass
        else:
            pass
        X = np.array(X, dtype=np.dtype("float32"))
        X = np.reshape(X, (X.shape[0], 1, X[0].shape[0], X[0].shape[1]))
        X = (X + config['offset']) / config['max']
        clips = [[] for i in range(2)]
        minibatch_size = config['minibatch_size']
        input_seq_length = config['input_seq_length']
        output_seq_length = config['output_seq_length']

        if self.mode == 'train': 
            for x in range(X.shape[0] / (input_seq_length + output_seq_length)):
                clips[0].append(
                    [(input_seq_length + output_seq_length) * x, input_seq_length])
                clips[1].append([(input_seq_length + output_seq_length)
                                 * x + input_seq_length, output_seq_length])
        elif self.mode == 'predict':
            for x in range(X.shape[0] / (input_seq_length)):
                clips[0].append(
                    [(input_seq_length + output_seq_length) * x, input_seq_length])
                clips[1].append(
                    [(input_seq_length + output_seq_length) * x + input_seq_length, 0])
        clips = np.array(clips, dtype=np.dtype("int32"))
        dims = np.array([[1, X[0].shape[1], X[0].shape[2]]],
                        dtype=np.dtype("int32"))
        return {'input_raw_data': X, 'dims': dims, "clips": clips}


def load_data(config, mode='predict'):

    iterator = WeatherIterator(config, mode=mode)
    iterator.begin(do_shuffle=True)
    #load_lg.info('Load done!')
    return iterator
