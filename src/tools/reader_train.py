# -*- coding: utf-8 -*-
# Author:Joe-BU
# Date: 2019-03-11

import sys
sys.path.append("../")
from settings import *
from sparnn.iterators import DataIterator
import numpy as np
from utils import *
from multiprocessing import Pool, Process
import theano


class WeatherIterator(DataIterator):
    def __init__(self, iterator_param, p0, p1, p2, p3, p4, mode='train'):
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
        self.concat_part(p0, p1, p2, p3, p4)

    def concat_part(self, part0, part1, part2, part3, part4):
        if self.mode == 'train':
            all_data = np.vstack([part0.data['input_raw_data'], part1.data['input_raw_data'],
                                  part2.data['input_raw_data'], part3.data['input_raw_data'],
                                  part4.data['input_raw_data']])

            self.data['input_raw_data'] = all_data
            self.data['dims'] = np.array([[1, all_data[0].shape[1], all_data[0].shape[2]]],
                                         dtype=np.dtype("int32"))

            clips = [[] for i in range(2)]
            for x in range(all_data.shape[0] /
                           (self.config['input_seq_length'] + self.config['output_seq_length'])):
                clips[0].append([(self.config['input_seq_length'] + self.config['output_seq_length']) * x,
                                 self.config['input_seq_length']])
                clips[1].append([(self.config['input_seq_length'] + self.config['output_seq_length']) * x +
                                 self.config['input_seq_length'], self.config['output_seq_length']])
            clips = np.array(clips, dtype=np.dtype("int32"))
            print clips
            self.data['clips'] = clips

            self.check_data()


def load_data1(config, part0, part1, part2, part3, part4, mode='predict'):
    iterator = WeatherIterator(config, part0, part1, part2,
                               part3, part4, mode=mode)
    iterator.begin(do_shuffle=True)
    return iterator
