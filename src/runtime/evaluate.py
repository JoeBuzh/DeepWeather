# -*- coding: utf-8 -*-
# Author:Joe-BU
# Date: 2019-04-09


import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
import argparse
pd.set_option("display.width", 1000)
from MaxDataEvaluator import MaxDataEvaluator
from model_config import base_config


def check_dir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)


def parse_args():
    parser = argparse.ArgumentParser(description='Get evaluate params')
    parser.add_argument('start', type=str, help='evaluate start time')
    parser.add_argument('end', type=str, help='evaluate end time')
    parser.add_argument('method', type=str, default='1x1', help='1x1 or 5x5')
    parser.add_argument('startX', type=int, default=0,
                        help='region left limit')
    parser.add_argument('endX', type=int, default=799,
                        help='region right limit')
    parser.add_argument('startY', type=int, default=0, help='region top limit')
    parser.add_argument('endY', type=int, default=799,
                        help='region down limit')

    args = parser.parse_args()
    start = args.start
    end = args.end
    method = args.method
    startX = args.startX
    endX = args.endX
    startY = args.startY
    endY = args.endY

    return start, end, method, startX, endX, startY, endY


def main():
    # change args in command line
    '''
    args = parse_args()
    start = datetime.strptime(args[0], "%Y%m%d%H%M")
    end = datetime.strptime(args[1], "%Y%m%d%H%M")
    method = args[2]
    startX = args[3]
    endX = args[4]
    startY = args[5]
    endY = args[6]
    '''

    # change args in model_config
    start = base_config['evaluate_start']
    end = base_config['evaluate_end']
    method = base_config['evaluate_method']
    startX = base_config['startX']
    endX = base_config['endX']
    startY = base_config['startY']
    endY = base_config['endY']

    Flag = True
    while start <= end:
        data = [[] for i in range(2)]
        config = {'threshold': base_config['threshold_list'],
                  'time': start.strftime("%Y%m%d%H%M"),
                  'root': base_config['to_path'],
                  'real_path': os.path.join(base_config['to_path'],
                                            os.path.join(
                                                start.strftime("%Y%m")),
                                            start.strftime("%Y%m%d%H%M")),
                  'predict_path': None,
                  'savepath': os.path.join(base_config['to_path'],
                                           os.path.join(start.strftime("%Y%m"), 'evaluate_record')),
                  'method': method,
                  'startX': startX,
                  'endX': endX,
                  'startY': startY,
                  'endY': endY
                  }

        print("Evaluating: " + start.strftime("%Y-%m-%d %H:%M"))
        check_dir(config['savepath'])
        if os.path.exists(os.path.join(config['real_path'], start.strftime("%Y%m%d%H%M") + ".json")):
            # print os.path.join(config['real_path'], start.strftime("%Y%m%d%H%M") + '.json')
            real_file = os.path.join(
                config['real_path'], start.strftime("%Y%m%d%H%M") + ".json")
            with open(real_file, "r") as f:
                content = json.load(f)
                data_part = content[-1]
                data[0].append(np.reshape(
                    np.array(data_part['data']), (800, 800)))
        else:
            print(" Real date is missing ! ")
            break

        # i = 0
        if not Flag:
            for i in range(base_config['output_seq_length']):
                target = start - \
                    timedelta(minutes=base_config['interval'] * (i + 1) * 2)
                target_dir = os.path.join(
                    config['root'], target.strftime("%Y%m") + '/' + target.strftime("%Y%m%d%H%M"))
                target_file = os.path.join(target_dir, target.strftime(
                    "%Y%m%d%H%M") + '-' + str(i) + '.json')
                if os.path.exists(target_file):
                    # print target_file
                    with open(target_file, 'r') as f:
                        content = json.load(f)
                        data_part = content[-1]
                        data[1].append(np.reshape(
                            np.array(data_part['data']), (800, 800)))
                else:
                    print(target_file + ' does not exist!')
                    break
        else:
            for i in range(4):
                target = start - \
                    timedelta(minutes=base_config['interval'] * (i + 1) * 2)
                target_dir = os.path.join(
                    config['root'], target.strftime("%Y%m") + '/' + target.strftime("%Y%m%d%H%M"))
                for j in range(3):
                    target_file = os.path.join(target_dir, target.strftime(
                        "%Y%m%d%H%M") + '-' + str(j) + '.json')
                    if os.path.exists(target_file):
                        # print target_file
                        with open(target_file, 'r') as f:
                            content = json.load(f)
                            data_part = content[-1]
                            data[1].append(np.reshape(
                                np.array(data_part['data']), (800, 800)))
                    else:
                        print(target_file + ' does not exist!')
                        break
        data[1] = [data[1][0], data[1][3], data[1][6], data[1][1], data[1][4], data[1][7],
                   data[1][2], data[1][9], data[1][10], data[1][5], data[1][8], data[1][11]]
        print(len(data))
        print(len(data[0]))
        print(len(data[1]))

        evaluator = MaxDataEvaluator(data=data, config=config)
        evaluator.merge()
        start = start + timedelta(minutes=base_config['interval'] * base_config['interval_times'])


if __name__ == '__main__':
    main()
