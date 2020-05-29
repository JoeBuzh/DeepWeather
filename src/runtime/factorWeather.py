# -*- coding: utf-8 -*-

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

#from convlstm.iterators import NumpyIterator
from sparnn.layers import InterfaceLayer
from sparnn.layers import AggregatePoolingLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import ConvLSTMLayer
from sparnn.layers import ConvForwardLayer
#from convlstm.layers import CombineLayer
from sparnn.layers import ConvRNNLayer
from sparnn.layers import ElementwiseCostLayer
from sparnn.layers import EmbeddingLayer
from sparnn.layers import GenerationLayer
from sparnn.models import Model

from sparnn.optimizers import SGD
from sparnn.optimizers import RMSProp
from sparnn.optimizers import AdaDelta

#from reader import load_data, get_config

import os
import time
import datetime
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

random.seed(1000)
numpy.random.seed(1000)

rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng()


class FactorWeatherModel(Model):
    def __init__(self, param, config):

        self.param = param
        self.config = config
        # 2x2 小碎片是2*2的小图片（叫做patch，或者掩模）
        feature_num = config['patch_size'] * config['patch_size']
        row_num = int(config['size'][0] / config['patch_size'])
        col_num = int(config['size'][1] / config['patch_size'])
        data_dim = (feature_num, row_num, col_num)
        kernel_size = config['kernel_size']
        kernel_num = config['kernel_num']

        logger.debug('build model:' + str(feature_num) + ' ' +
                     str(data_dim) + ' ' + str(kernel_size) + ' ' + str(kernel_num))
        param = {"id": "DeepWeather-interface",
                 "use_input_mask": False, "input_ndim": 5, "output_ndim": 5}
        self.interface_layer = InterfaceLayer(param)

        # interface_layer.input: (1, 100, 100), reshape_input: (16, 25, 25),  data_dim: (16, 25, 25)
        self.reshape_input = quick_reshape_patch(
            self.interface_layer.input, config['patch_size'])
        self.reshape_output = quick_reshape_patch(
            self.interface_layer.output, config['patch_size'])

        minibatch_size = self.interface_layer.input.shape[1]

        self.middle_layers = []

        param = {"id": 0, "rng": rng, "theano_rng": theano_rng, "dim_in": data_dim,
                 "dim_out": (kernel_num[0], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "minibatch_size": minibatch_size,
                 "input": self.reshape_input,
                 "n_steps": config['input_seq_length']}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 1, "rng": rng, "theano_rng": theano_rng, "dim_in": (kernel_num[0], row_num, col_num),
                 "dim_out": (kernel_num[1], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "minibatch_size": minibatch_size,
                 "input": self.middle_layers[0].output,
                 "n_steps": config['input_seq_length']}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 2, "rng": rng, "theano_rng": theano_rng, "dim_in": (kernel_num[1], row_num, col_num),
                 "dim_out": (kernel_num[2], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "minibatch_size": minibatch_size,
                 "input": self.middle_layers[1].output,
                 "n_steps": config['input_seq_length']}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 3, "rng": rng, "theano_rng": theano_rng, "dim_in": data_dim,
                 "dim_out": (kernel_num[0], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "init_hidden_state": self.middle_layers[0].output[-1],
                 "init_cell_state": self.middle_layers[0].cell_output[-1],
                 "minibatch_size": minibatch_size,
                 "input": None,
                 "n_steps": config['output_seq_length'] - 1}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 4, "rng": rng, "theano_rng": theano_rng, "dim_in": (kernel_num[0], row_num, col_num),
                 "dim_out": (kernel_num[1], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "init_hidden_state": self.middle_layers[1].output[-1],
                 "init_cell_state": self.middle_layers[1].cell_output[-1],
                 "minibatch_size": minibatch_size,
                 "input": self.middle_layers[3].output,
                 "n_steps": config['output_seq_length'] - 1}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 5, "rng": rng, "theano_rng": theano_rng, "dim_in": (kernel_num[1], row_num, col_num),
                 "dim_out": (kernel_num[2], row_num, col_num),
                 "input_receptive_field": (kernel_size, kernel_size),
                 "transition_receptive_field": (kernel_size, kernel_size),
                 "init_hidden_state": self.middle_layers[2].output[-1],
                 "init_cell_state": self.middle_layers[2].cell_output[-1],
                 "minibatch_size": minibatch_size,
                 "input": self.middle_layers[4].output,
                 "n_steps": config['output_seq_length'] - 1}
        self.middle_layers.append(ConvLSTMLayer(param))

        param = {"id": 6, "rng": rng, "theano_rng": theano_rng,
                 "dim_in": (kernel_num[0] + kernel_num[1] + kernel_num[2], row_num, col_num),
                 "dim_out": data_dim, "input_receptive_field": (1, 1),
                 "input_stride": (1, 1), "activation": "sigmoid",
                 "minibatch_size": minibatch_size,
                 "conv_type": "same",
                 "input": TT.concatenate([
                     TT.concatenate([
                         self.middle_layers[0].output[-1:],
                         self.middle_layers[1].output[-1:],
                         self.middle_layers[2].output[-1:]], axis=2),
                     TT.concatenate([
                         self.middle_layers[3].output,
                         self.middle_layers[4].output,
                         self.middle_layers[5].output], axis=2)])
                 }
        self.middle_layers.append(ConvForwardLayer(param))

        # dim_in = (16, 25, 25)
        param = {"id": "cost", "rng": rng, "theano_rng": theano_rng,  # "cost_func": "BinaryCrossEntropy",
                 "cost_func": config['cost_func'],
                 "dim_in": data_dim, "dim_out": (1, 1, 1),
                 "minibatch_size": minibatch_size,
                 "input": self.middle_layers[-1].output,
                 "target": self.reshape_output}
        self.cost_layer = ElementwiseCostLayer(param)
        self.outputs = [{"name": "prediction",
                         "value": self.middle_layers[-1].output}]

        model_param = {'interface_layer': self.interface_layer, 'middle_layers': self.middle_layers, 'cost_layer': self.cost_layer,
                       #'combine_interface_layer': self.combine_interface_layer,
                       'outputs': self.outputs, 'errors': None,
                       'name': "DeepWeatherModel-test",
                       'problem_type': "regression"}
        super(FactorWeatherModel, self).__init__(model_param)


def test():
    train_iterator, valid_iterator, test_iterator = load_data()

    test_config = get_config('test')
    print('configuration:', test_config)

    param = {'errors': None,
             'name': "Moving-MNIST-Model-Convolutional-test-unconditional",
             'problem_type': "regression"}
    model = DeepWeatherModel(param, test_config)

    param = {'id': '1', 'learning_rate': test_config['learning_rate'], 'momentum': 0.9, 'decay_rate': 0.9, 'clip_threshold': None,
             'max_epoch': test_config['max_epoch'], 'start_epoch': 0, 'max_epochs_no_best': 200, 'decay_step': 200,
             'autosave_mode': ['interval', 'best'], 'save_path': test_config['save_path'], 'save_interval': 30}
    optimizer = RMSProp(model, train_iterator,
                        valid_iterator, test_iterator, param)

    optimizer.observe(test_config['patch_size'])
    #optimizer.predict(test_config, save_path)


if __name__ == "__main__":
    model_path = '/home/guodongchen/weather/model'
    model_name = 'model-proceed-radar.pkl'
    print('Load model:', model_path, model_name)
    model = FactorWeatherModel.load(os.path.join(model_path, model_name))
