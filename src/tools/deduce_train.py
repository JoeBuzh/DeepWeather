'''
zboleibj@cn.ibm.com
Joe-bu@i2value.com
'''
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from settings import *
import os
import os.path
import random
from multiprocessing import Pool, Process

import theano

from utils import *
from reader import *
from reader_train import *
from factorWeather import FactorWeatherModel

import sparnn
from sparnn.optimizers import RMSProp

import argparse
import numpy as np
import datetime
import time
import logging
import ConfigParser
from netCDF4 import Dataset
from crypt import crypt
from mylog import mylog as lg

#sparnn.utils.quick_logging_config('deep-weather.log')

def float_formatter(x): return "%.2f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
# setting theano configs
# Enable or disable parallel computation on the CPU with OpenMP
theano.config.openmp = True
theano.config.optimizer = "fast_compile"
# The number of user stack level to keep for variables.
theano.config.traceback.limit = 100

my_log = lg.init_logger(deduce_log_path)

radar_config = {
    'name': 'radar',
    'max': 14.0,  
    'offset': 0.0,
    'cmap': 'radar',
    'cost_func': 'Fade',
    'level': 0,
    'data_dir': radar_data_dir,
    'save_dir': radar_save_dir
}
wind_config = {
    'name': 'wind',
    'level': 10,
    'max': 50.0,
    'offset': 25.0,  # x=(x+offset)/max
    'cmap': 'wind',
    'data_dir': wind_data_dir,
    'save_dir': wind_save_dir,
    'cost_func': 'BinaryCrossEntropy'
}
pgm_config = {
    'name': 'pgm',
    'level': 10,
    'max': 255.0,
    'offset': 0,  # x=(x+offset)/max
    'cmap': 'pgm',
    'data_dir': pgm_data_dir,
    'save_dir': pgm_save_dir,
    'cost_func': 'Fade'
}

model_config = {
    'interval': 6, 
    'input_seq_length': 10,
    'output_seq_length': 20,
    'minibatch_size': 8,
    'learning_rate': 0.003,  # ori 0.002
    'patch_size': 2,
    'max_epoch': 20,  
    'layer_num': 6,
    'layer_name': 'lstm',
    'kernel_size': 3,
    'kernel_num': (64, 64, 64),

    'size': (200, 200), 
    'compress': 2,


    'model_path': model_path,
    #'vmax': 100,#instead of 'model'
    'use_input_mask': False,
    'input_data_type': 'float32',
    'is_output_sequence': True,
    #'name': 'DeepWeather',
    'cost_func': 'Fade',
    #'cost_func': 'BinaryCrossEntropy'  #
}


def check_crypt():
    with open('crypt') as f:
        line = f.readline()
        if line == crypt():
            return True
    return False



def get_config(mode='train', src='radar'):
    if src == 'radar':
        config = dict(model_config, **radar_config)
    elif src == 'wind':
        config = dict(model_config, **wind_config)
    elif src == 'pgm':
        config = dict(model_config, **pgm_config)

    if mode == "train":
        config['start_date'] = start_date
        config['end_date'] = end_date
    elif mode == "valid":
        config['start_date'] = valid_start_date
        config['end_date'] = valid_end_date
    elif mode == "test":
        config['start_date'] = test_start_date
        config['end_date'] = test_end_date

    return config


def gen_train(cfg, mode, split_num, piece_num):
    #print cfg['start_date']
    #print cfg['end_date']
    duration = cfg['end_date'] - cfg['start_date']
    #print('duration:', duration)
    #print duration.days
    #print duration.seconds
    duration_second = duration.days*24*3600 + duration.seconds
    #print duration_second
    pieces = duration_second / 360
    #print pieces
    piece = pieces / split_num
    #print piece
    cfg['start_date'] = cfg['start_date'] + datetime.timedelta(minutes=piece_num * piece * 6)
    #print cfg['start_date']
    if piece_num == 4:
        pass
    else:
        cfg['end_date'] = cfg['start_date'] + datetime.timedelta(minutes=piece *6)
    #print cfg['start_date']
    #print cfg['end_date']
    train_iterator = load_data(cfg, mode)
    return train_iterator


def gen_valid(cfg, mode):
    valid_iterator = load_data(cfg, mode)
    return valid_iterator


def gen_test(cfg, mode):
    test_iterator = load_data(cfg, mode)
    return test_iterator


def train(src='radar'):
    random.seed(1000)
    np.random.seed(1000)
    rng = sparnn.utils.quick_npy_rng()
    theano_rng = sparnn.utils.quick_theano_rng()

    config = get_config('train', src=src)
    valid_config = get_config('valid', src=src)
    test_config = get_config('test', src=src)

    # init the model, change minibatch size to tensor value
    param = {'errors': None,
             'name': "FactorWeather-Convolutional-cloud",
             'problem_type': "regression"}
    model = FactorWeatherModel(param, config)

    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])

    my_log.info(' Start load data ...')
    pool = Pool(processes=7)

    train_p0 = pool.apply_async(func=gen_train, args=(config, 'train', 5, 0))
    train_p1 = pool.apply_async(func=gen_train, args=(config, 'train', 5, 1))
    train_p2 = pool.apply_async(func=gen_train, args=(config, 'train', 5, 2))
    train_p3 = pool.apply_async(func=gen_train, args=(config, 'train', 5, 3))
    train_p4 = pool.apply_async(func=gen_train, args=(config, 'train', 5, 4))

    valid_p = pool.apply_async(func=gen_valid, args=(valid_config, 'train'))
    test_p = pool.apply_async(func=gen_test, args=(test_config, 'train'))

    pool.close()
    pool.join()

    train_iterator0 = train_p0.get()
    train_iterator1 = train_p1.get()
    train_iterator2 = train_p2.get()
    train_iterator3 = train_p3.get()
    train_iterator4 = train_p4.get()

    train_iterator = load_data1(config, train_iterator0,
                                train_iterator1, train_iterator2,
                                train_iterator3, train_iterator4,
                                'train')
    valid_iterator = valid_p.get()
    test_iterator = test_p.get()

    my_log.info(' Load all data done!')
    my_log.info(train_iterator0.data['input_raw_data'].shape)
    my_log.info(train_iterator1.data['input_raw_data'].shape)
    my_log.info(train_iterator2.data['input_raw_data'].shape)
    my_log.info(train_iterator3.data['input_raw_data'].shape)
    my_log.info(train_iterator4.data['input_raw_data'].shape)
    my_log.info(train_iterator.data['input_raw_data'].shape)
    my_log.info(valid_iterator.data['input_raw_data'].shape)
    my_log.info(test_iterator.data['input_raw_data'].shape)
    # build the optimizer
    param = {'id': '1',
             'learning_rate': config['learning_rate'], 'momentum': 0.9, 'decay_rate': 0.9, 'clip_threshold': None,
             'max_epoch': config['max_epoch'],
             'start_epoch': 0, 'max_epochs_no_best': 200, 'decay_step': 200,
             'autosave_mode': ['interval', 'best', 'proceed'],
             'save_path': config['model_path'], 'save_interval': 30,
             'model_name': 'model-proceed-' + config['name'] + '.pkl'}
    try:
        print('begin optimizer...')
        optimizer = RMSProp(model, train_iterator,
                            valid_iterator, test_iterator, param)
        print("optimizer object finished,begin train ")
        optimizer.train(config, config['model_path'])
    except Exception as e:
        my_log.info('error1: ' + str(e))


def predict(begin_date, end_date, src='radar'):
    config = get_config('test', src=src)
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # prepare model
    model_path = config['model_path']
    model_name = 'DeepWeatherModel-test-validation-best.pkl'
    model = FactorWeatherModel.load(os.path.join(model_path, model_name))
    # print("done")
    model.set_mode("predict")
    predict_func = theano.function(inputs=model.interface_layer.input_symbols(),
                                   outputs=sparnn.utils.quick_reshape_patch_back(model.middle_layers[-1].output,
                                                                                 config['patch_size']),
                                   on_unused_input='ignore')

    it = begin_date
    while(it < end_date):  # original while(it <= end_time):
        it += datetime.timedelta(minutes=6)

        # get results
        start_date = it - \
            datetime.timedelta(
                minutes=(config['input_seq_length'] - 1) * config['interval'])

        # load data and check input
        config['start_date'] = start_date
        config['end_date'] = it
        # print('loading data', config['start_date'], config['end_date'])
        try:
            test_iterator = load_data(config, mode='predict')
            test_iterator.begin(do_shuffle=False)
        except Exception as e:
            print(Exception, e)
            continue

        result = predict_func(*(test_iterator.input_batch())) * \
            config['max'] - config['offset']
        # print(result)

        # get the result dir
        result_dir = os.path.join(save_dir, it.strftime('%Y%m'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # save the input dir
        input_image = np.reshape(test_iterator.input_batch()[
                                 0][-1][0], (1, config['size'][0], config['size'][1]))[0] * config['max'] - config['offset']
        write_image(input_image, result_dir, it, config)

        print('predict', it, result.shape, input_image.max(),
              input_image.min(), result.max(), result.min())
        # save the result files
        for i, r in enumerate(result):
            image = np.reshape(
                r[0], (1, config['size'][0], config['size'][1]))[0]
            write_image(image, result_dir, it, config, predict=i)


def run(src='radar'):
    # if not check_crypt():
    #    return

    config = get_config('test', src=src)
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # prepare model
    model_path = config['model_path']
    model_name = 'model-proceed-' + config['name'] + '.pkl'
    model = FactorWeatherModel.load(os.path.join(model_path, model_name))
    model.set_mode("predict")
    predict_func = theano.function(inputs=model.interface_layer.input_symbols(),
                                   outputs=sparnn.utils.quick_reshape_patch_back(model.middle_layers[-1].output,
                                                                                 config['patch_size']),
                                   on_unused_input='ignore')

    last_predict = None
    while(True):
        it = datetime.datetime.now()
        start_date = it - \
            datetime.timedelta(
                minutes=(config['input_seq_length'] - 1) * config['interval'])

        # load data and check input
        config['start_date'] = start_date
        config['end_date'] = it
        if last_predict == None or last_predict.hour != it.hour:
            try:
                test_iterator = load_data(config, mode='predict')
                test_iterator.begin(do_shuffle=False)
            except Exception as e:  # no data loaded
                print('missing data')
                time.sleep(5 * 60)
                continue
            last_predict = it
        else:
            print('Only predict once each hour', last_predict)
            time.sleep(5 * 60)
            continue
        result = predict_func(*(test_iterator.input_batch())) * \
            config['max'] - config['offset']

        # get the result dir
        result_dir = os.path.join(save_dir, it.strftime('%Y%m'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # save the result data
        input_image = np.reshape(test_iterator.input_batch()[
                                 0][-1][0], (1, config['size'][0], config['size'][1]))[0] * config['max'] - config['offset']
        print(input_image)
        write_image(input_image, result_dir, it, config)

        # save the result files
        data = {}
        for i, r in enumerate(result):
            image = np.reshape(
                r[0], (1, config['size'][0], config['size'][1]))[0]
            write_image(input_image, result_dir, it, config, predict=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deduce future images.')
    parser.add_argument('mode', metavar='base', type=str, default='run',
                        help='Mode: run, train')
    parser.add_argument('--src', type=str, default='radar', required=False,
                        help='Type of data: radar, pgm, wind')

    args = parser.parse_args()
    mode = args.mode
    src = args.src

    if mode == 'train':
        try:
            train(src=src)
            my_log.info('End of the training')
        except Exception as e:
            my_log.info('error2: ' + str(e))

    elif mode == 'predict':
        begin_date = predict_begin
        end_date = predict_end
        predict(begin_date, end_date, src=src)

    elif mode == 'run':
        run(src=src)
