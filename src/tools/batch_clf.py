# -*- coding: utf-8 -*-
# Author: Joe BU
# Date: 2019-2-20 14:00


import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

pd.set_option('max_columns', 15)

plt.switch_backend('agg')


def merge_csv(begin, end, path1):
    clips = [[] for i in range(20)]
    while (begin <= end):
        filename = path1 + begin.strftime("%Y%m%d%H%M") + "_eva_result.csv"
        df = pd.read_csv(filename, engine='python', encoding='utf-8')
        for i in range(20):
            df1 = df.loc[df['num'] == (i + 1), :]
            clips[i].append(df1)
        begin = begin + timedelta(minutes=30)

    all_list = []
    for i in range(20):
        slices = pd.concat(clips[i])
        for i in range(len(slices['now_time'])):
            #             print(slices['now_time'].values[i])
            slices['now_time'].values[i] = datetime.strptime(
                slices['now_time'].values[i], '%Y-%m-%d %H:%M:00').strftime("%Y%m%d %H:%M")
#         print(slices)
        ploter(slices)
        all_list.append(slices)
    df_all = pd.concat(all_list)
    df_all.to_csv("/data/buzehao/test/" +
                  datetime.strptime(df_all['now_time'].values[0], '%Y%m%d %H:%M').strftime("%Y%m%d%H%M") +
                  "_all_result.csv")


def ploter(data):
    plt.figure(figsize=(25, 13))
    plt.plot(data['now_time'], data['POD'], label='POD index')
    plt.plot(data['now_time'], data['FAR'], label='FAR index')
    plt.plot(data['now_time'], data['CSI'], label='CSI index')
    plt.plot(data['now_time'], [0.5 * i / i for i in range(1,
                                                           data.shape[0] + 1)], 'r--', label='Standard')
    plt.xticks(rotation=50)
    # plt.xticks([data['TIMESTAMP'][i] for i in range(data[['TIMESTAMP']].shape[0]) if i % 5 == 0])
    # plt.xlabel('Time(min)', fontsize=16)
    # plt.ylabel('Value', rotation=True, fontsize=16, labelpad=15)
    plt.tick_params(labelsize=13)
    plt.title('预测第' + str(data['num'].unique()[0]) + '张图片评价指标变化',
              fontproperties="SimHei",
              fontsize=25)
    plt.legend(fontsize=18, loc='best')
    plt.grid(True)
    plt.savefig("/data/buzehao/test/" +
                datetime.strptime(data['now_time'].values[0], '%Y%m%d %H:%M').strftime("%Y%m%d%H%M") +
                "开始预测临近第" + str(data['num'].unique()[0]) + "张图片评价变化.png")


def check_dir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)


def main():
    csv_path = '/data/buzehao/weather_output20/save/radar/'
    check_dir('/data/buzehao/test')
    begin = datetime(2018, 6, 22, 0, 0, 0)
    end = datetime(2018, 6, 23, 0, 0, 0)
    merge_csv(begin, end, csv_path)


if __name__ == '__main__':
    main()
