# -*- coding: utf-8 -*-
# Author:Joe-BU
# Date:2019-4-29

from datetime import datetime, timedelta


base_config = {
    'wind_data_dir': './tmp/',
    'radar_data_dir': '/home/lenon/radar/',
    'pgm_data_dir': '../home/pgm',

    'radar_save_dir': '/home/lenon/save/radar',
    'wind_save_dir': './save/wind',
    'pgm_save_dir': '../save/pgm',

    'start_date0': datetime(2018, 6, 1, 0, 0, 0),
    'end_date0': datetime(2018, 6, 2, 23, 54, 0),

    'start_date1': datetime(2018, 6, 3, 0, 0, 0),
    'end_date1': datetime(2018, 6, 4, 23, 54, 0),

    'start_date2': datetime(2018, 6, 5, 0, 0, 0),
    'end_date2': datetime(2018, 6, 6, 23, 54, 0),

    'start_date3': datetime(2018, 6, 7, 0, 0, 0),
    'end_date3': datetime(2018, 6, 8, 23, 54, 0),

    'start_date4': datetime(2018, 6, 9, 0, 0, 0),
    'end_date4': datetime(2018, 6, 10, 23, 54, 0),

    'start_date5': datetime(2018, 6, 11, 0, 0, 0),
    'end_date5': datetime(2018, 6, 12, 23, 54, 0),

    'valid_start_date': datetime(2018, 6, 13, 0, 0, 0),
    'valid_end_date': datetime(2018, 6, 14, 23, 54, 0),

    'test_start_date': datetime(2018, 6, 15, 0, 0, 0),
    'test_end_date': datetime(2018, 6, 16, 23, 54, 0),

    'predict_begin': datetime(2018, 7, 7, 10, 23, 0),
    'predict_end': datetime(2018, 7, 7, 18, 3, 0),

    # radar model params
    'radar_max': 14.0,
    'radar_offset': 0.0,
    'compress': 4,
    'size': (200, 200),
    'interval': 12,
    'input_seq_length': 5,
    'output_seq_length': 10,
    'minibatch_size': 8,
    'max_epoch': 20,
    'learning_rate': 0.003,
    'use_input_mask': False,
    'train_set_num': 6,

    # path
    'append_path': '/home/lenon/weather_update/src/',
    'deduce_log_path': '/home/lenon/logs_max/train-process/deduce-main.log',
    'load_log_path': '/home/lenon/logs_max/train-process/load-data-detail.log',
    'process_log_path': '/home/lenon/logs_max/train-process/process.log',
    'train_log_path': '/home/lenon/logs_max/train-process/train-detail.log',
    'preprocess_log_path': '/home/lenon/logs_max/prepare_train_data-process/preprocess.log',
    'train_pool': '/home/lenon/train_pool/',
    'runtime_log_path': '/home/lenon/logs_max/runtime-process/runtime.log',
    'remove_log_path': '/home/lenon/logs_max/remove-process/remove.log',
    'base_map_path': '/home/lenon/weather_update/src/../',
    'model_path': '/home/lenon/weather_update/model',
    'noise_path': '/home/lenon/weather_update/src/runtime',

    # for predict
    'raw_image_path': '/home/lenon/raw_picdata/',
    'to_path': '/home/lenon/predict/',
    # for evaluate
    'threshold_list': [5, 15, 30, 45, 55, 65],
    'evaluate_start': datetime(2018, 6, 30, 5, 33, 0),
    'evaluate_end': datetime(2018, 6, 30, 5, 33, 0),
    'evaluate_method': '5x5',
    'startX': 0,
    'endX': 799,
    'startY': 0,
    'endY': 799
}
