# -*- coding: utf-8 -*-
# Author:Joe-BU

from datetime import datetime, timedelta

startX, endX = 0, 799
startY, endY = 0, 799

wind_data_dir = './tmp/'
radar_data_dir = '/root/my/train_max/'
pgm_data_dir = '../data/pgm'

radar_save_dir = '/root/my/weather_max/save/radar'
wind_save_dir = './save/wind'
pgm_save_dir = '../save/pgm'
# ------------------------------------------------------------------
start_date = datetime(2018, 6, 1, 0, 3, 0)
end_date = datetime(2018, 6, 1, 23, 58, 0)

valid_start_date = datetime(2018, 6, 11, 0, 3, 0)
valid_end_date = datetime(2018, 6, 11, 12, 58, 0)

test_start_date = datetime(2018, 6, 13, 0, 3, 0)
test_end_date = datetime(2018, 6, 13, 12, 58, 0)

predict_begin = datetime(2018, 6, 20, 10, 13, 0)
predict_end = datetime(2018, 6, 20, 10, 18, 0)
# ------------------------------------------------------------------
append_path = '/root/my/weather_max/src/'
deduce_log_path = '/root/my/logs/train-process/deduce-main.log'
load_log_path = '/root/my/logs/train-process/load-data-detail.log'
train_log_path = '/root/my/logs/train-process/train-detail.log'
preprocess_log_path = '/root/my/logs/prepare_train_data-process/preprocess.log'
train_pool = '/root/my/train_pool/'
runtime_log_path = '/root/my/logs/runtime-process/runtime.log'
remove_log_path = '/root/my/logs/remove-process/remove.log'
# ------------------------------------------------------------------
model_path = '/root/my/weather_max/model'
base_map_path = '/root/my/weather_max/src/NE_base_new.gif'
# ------------------------------------------------------------------
# for predict and evaluate
raw_image_path = '/root/my/Image/raw_picdata/'
to_path = '/root/my/predict_data/'
