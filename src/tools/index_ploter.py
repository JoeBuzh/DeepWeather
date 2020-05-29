# -*- coding: utf-8 -*-
# Author: Joe BU
# Date: 2019-2-13

import time
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from settings import *

import sys
reload(sys)
sys.setdefaultencoding('gb18030') 

import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('agg')
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def index_plot(data):
    plt.figure(figsize=(16, 12))
    plt.plot(data['pre_time'], data['POD'], label='POD index')
    plt.plot(data['pre_time'], data['FAR'], label='FAR index')
    plt.plot(data['pre_time'], data['CSI'], label='CSI index')
    plt.plot(data['pre_time'],
             [0.5 * i / i for i in range(1, data.shape[0] + 1)],
             'r--',
             label='Standard')
    plt.xticks(rotation=50)
    plt.tick_params(labelsize=12)
    plt.title(data['now_time'].unique()[0][:-3] + ' Predict Performance',
              fontproperties="SimHei",
              fontsize=25)
    plt.legend(fontsize=18, loc='best')
    plt.grid(True)
    plt.savefig("/data/buzehao/temp/" + sys.argv[1] + '/' + datetime.strptime(data['now_time'].unique()[
                0], '%Y-%m-%d %H:%M:00').strftime("%Y%m%d%H%M") + "_evaluate.png")


def main():
    df = pd.read_csv(os.path.join(radar_save_dir, sys.argv[1] + '_eva_result.csv'),
                     encoding='utf-8')
    #df['pre_time'] = datetime.strptime(df['pre_time'], '%Y-%m-%d %H:%M:00').strftime("%Y%m%d%H%M")
    for i in range(len(df['pre_time'])):
        df['pre_time'][i] = datetime.strptime(df['pre_time'][i], '%Y-%m-%d %H:%M:00').strftime("%Y%m%d %H:%M")
    index_plot(data=df)


if __name__ == '__main__':
    main()
