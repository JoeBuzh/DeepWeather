# -*- coding: utf-8 -*-
# Author:Joe BU
# Date:2019-2-25 11:00

from settings import *
import sys
sys.path.append(append_path)
import os
import os.path
import random
import theano
import json
from utils import *
from reader import *
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

# sparnn.utils.quick_logging_config('deep-weather.log')


def float_formatter(x): return "%.2f" % x


np.set_printoptions(formatter={'float_kind': float_formatter})
# Enable or disable parallel computation on the CPU with OpenMP
theano.config.openmp = True
theano.config.optimizer = "fast_compile"
# The number of user stack level to keep for variables.
theano.config.traceback.limit = 100

my_log = lg.init_logger(deduce_log_path)
# dictMerged2=dict(dict1, **dict2)

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
    'offset': 25.0,
    'cmap': 'wind',
    'data_dir': wind_data_dir,
    'save_dir': wind_save_dir,
    'cost_func': 'BinaryCrossEntropy'
}
pgm_config = {
    'name': 'pgm',
    'level': 10,
    'max': 255.0,
    'offset': 0,
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
    'model_name': None,
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


def predict(begin_date, end_date, save_mode, src='radar'):
    config = get_config('test', src=src)
    config['savejson'] = save_mode
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = config['model_path']
    model_list = ['DeepWeatherModel_5690.pkl', 'DeepWeatherModel_800.pkl']
    predict_res = [[] for i in range(len(model_list) + 1)]
    for x, model_name in enumerate(model_list):
        print('Load model:' + model_path + '/' + model_name)
        config['model_name'] = model_name
        model = FactorWeatherModel.load(os.path.join(model_path, model_name))
        print("done")
        model.set_mode("predict")
        predict_func = theano.function(inputs=model.interface_layer.input_symbols(),
                                       outputs=sparnn.utils.quick_reshape_patch_back(model.middle_layers[-1].output,
                                                                                     config['patch_size']),
                                       on_unused_input='ignore')

        it = begin_date
        while(it <= end_date):
            #it += datetime.timedelta(minutes=6)
            start_date = it - \
                datetime.timedelta(
                    minutes=(config['input_seq_length'] - 1) * config['interval'])

            config['start_date'] = start_date
            config['end_date'] = it
            print('loading data', config['start_date'], config['end_date'])
            try:
                test_iterator = load_data(config, mode='predict')
                test_iterator.begin(do_shuffle=False)
            except Exception as e:
                print(Exception, e)
                continue

            result = predict_func(*(test_iterator.input_batch())) * \
                config['max'] - config['offset']

            result_dir = os.path.join(save_dir, it.strftime('%Y%m%d%H%M'))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            input_image = np.reshape(test_iterator.input_batch()[
                                     0][-1][0], (1, config['size'][0], config['size'][1]))[0] * config['max'] - config['offset']
            #write_image_update(input_image, result_dir, it, config)

            print('predict', it, result.shape, input_image.max(),
                  input_image.min(), result.max(), result.min())
            for i, r in enumerate(result):
                image = np.reshape(
                    r[0], (1, config['size'][0], config['size'][1]))[0]
                image = resize(
                    image, (config['size'][0] * config['compress'], config['size'][1] * config['compress']))
                predict_res[x].append(image)
                write_image_update(image, result_dir, it, config, predict=i)
            it += datetime.timedelta(minutes=6)

    for i in range(config['output_seq_length']):
        # file1 = begin_date.strftime(
        #     "%Y%m%d%H%M") + "-" + str(i) + "-" + model_list[0][:-4] + '.json'
        # file2 = begin_date.strftime(
        #     "%Y%m%d%H%M") + "-" + str(i) + "-" + model_list[1][:-4] + '.json'
        # a1 = np.array(json.load(open(os.path.join(
        #     save_dir + "/" + begin_date.strftime("%Y%m%d%H%M"), file1))))
        # a2 = np.array(json.load(open(os.path.join(
        #     save_dir + "/" + begin_date.strftime("%Y%m%d%H%M"), file2))))
        a1 = predict_res[0][i]
        a2 = predict_res[1][i]
        A = 0.7 * a1 + 0.5 * a2
        config['savejson'] = 'both'
        predict_res[-1].append(A)
        # write_image(A, os.path.join(save_dir, begin_date.strftime("%Y%m%d%H%M")),
        #             begin_date, config, predict=i)
        write_image(predict_res[-1][i], os.path.join(save_dir, begin_date.strftime("%Y%m%d%H%M")),
                    begin_date, config, predict=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deduce future images.')
    parser.add_argument('mode', metavar='base', type=str, default='run',
                        help='Mode: run, train')
    parser.add_argument('--src', type=str, default='radar', required=False,
                        help='Type of data: radar, pgm, wind')
    parser.add_argument('save_mode', type=str, default='onlypng',
                        help='if save jsonfile: onlypng or both')
    parser.add_argument('start', type=str,
                        default='201902192000', help='predict start time')
    parser.add_argument('end', type=str,
                        default='201902192030', help='predict end time')

    args = parser.parse_args()
    mode = args.mode
    src = args.src
    save_mode = args.save_mode
    start = args.start
    end = args.end

    if mode == 'train':
        try:
            train(src=src)
            my_log.info('End of the training')
        except Exception as e:
            my_log.info('error2: ' + str(e))

    elif mode == 'predict':
        begin_date = datetime.datetime.strptime(start, "%Y%m%d%H%M")
        end_date = datetime.datetime.strptime(end, "%Y%m%d%H%M")
        predict(begin_date, end_date, save_mode, src=src)

    elif mode == 'run':
        run(src=src)
