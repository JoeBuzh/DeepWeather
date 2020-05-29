
import os, os.path
import random

import argparse
import numpy as np
import datetime, time
import logging
import ConfigParser

from utils import *

model_config = {
    'interval': 60,
    'minibatch_size': 8,
    'learning_rate': 0.002,
    'max_epoch': 20,

    'name': 'Total precipitation',
    'level': 0,
    'size': (220, 220),
    'model_path': 'model',
    'name': 'DeepWeather',
}

def get_config(mode='train', src='radar'):
    config = model_config.copy()

    config_name='config.ini'
    cf = ConfigParser.ConfigParser()
    cf.read(config_name)
    config['data_dir'] = cf.get('data', 'data_dir')
    config['save_dir'] = cf.get('data', 'save_dir')
    config['src'] = src


    if mode == "train":
        config['start_date'] = datetime.datetime(2017, 4, 1, 0, 0,  0)
        config['end_date']   = datetime.datetime(2017, 4, 30, 23, 59,  59)

    elif mode == "valid":
        config['start_date'] = datetime.datetime(2017, 4, 1, 0, 0,  0)
        config['end_date']   = datetime.datetime(2017, 4, 10, 23, 59,  59)
    elif mode == "test":
        config['start_date'] = datetime.datetime(2017, 7, 1, 0, 0,  0)
        config['end_date']   = datetime.datetime(2017, 7, 3, 23, 59,  59)

    return config

def train(src='ana'):
    # load configuration
    config = get_config('train', src=src)
    valid_config = get_config('valid', src=src)
    test_config = get_config('test', src=src)

    # load data
    filenames = load_range_filenames(config['start_date'], config['end_date'], config['data_dri'])
    X = []
    for filename in filenames:
        X_hour = read_grib2(filename, name=config['name'], level=config['level'])
        X.append(X_hour)
    lats, lons = read_latlons(filename)
    X = np.array(X, dtype=np.dtype("float32"))

    # load label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deduce future images.')
    parser.add_argument('mode', metavar='base', type=str, default='run',
                        help='Mode: run, train')
    parser.add_argument('--src', type=str, default='global_radar', required=False,
                        help='Type of data: global_radar, pgm, rain')


    args = parser.parse_args()
    mode = args.mode
    src = args.src


    if mode == 'train':
        train(src=src)
    elif mode == 'predict':
        begin_date = datetime.datetime(2017, 4, 1, 0, 0,  0)
        end_date = datetime.datetime(2017, 4, 30, 23, 0,  0)
        predict(begin_date, end_date)
