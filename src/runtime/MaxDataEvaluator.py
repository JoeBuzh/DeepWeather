# -*- coding: utf-8 -*-
# Athuor: Joe-BU
# Date: 2019-04-04


import numpy as np
import json
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from model_config import base_config
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)


class MaxDataEvaluator(object):
    """
    Method for anti-direction evaluate 
    """

    def __init__(self, data, config):
        self.real = data[0]
        self.predict = data[1]
        self.time = config['time']
        self.threshold = config['threshold']
        self.root = config['root']
        self.real_path = config['real_path']
        self.predict_path = config['predict_path']
        self.savepath = config['savepath']
        self.method = '1x1' if config['method'] == None else config['method']
        self.startX = config['startX'] if config['startX'] != None else 0
        self.endX = config['endX'] + \
            1 if config['endX'] != None else self.real[0].shape[0]
        self.startY = config['startY'] if config['startY'] != None else 0
        self.endY = config['endY'] + \
            1 if config['endY'] != None else self.real[0].shape[1]

    def get_radar_level(self, val, thd_l, thd_r):
        if val >= thd_l and val <= thd_r:
            return 1
        else:
            return 0

    def stas_1_plus_1(self, num, level_ind):
        assert self.real[0].shape[0] == self.predict[num].shape[0]
        assert self.real[0].shape[1] == self.predict[num].shape[1]
        hit, miss, fake, white = 0.0, 0.0, 0.0, 0.0

        if level_ind == len(self.threshold) - 1:
            thd_l = self.threshold[level_ind]
            thd_r = float("inf")
        else:
            thd_l = self.threshold[level_ind]
            thd_r = self.threshold[level_ind + 1]

        for i in range(self.startX, self.endX):
            for j in range(self.startY, self.endY):
                if self.get_radar_level(self.real[0][i][j], thd_l, thd_r) == 1 and \
                        self.get_radar_level(self.predict[num][i][j], thd_l, thd_r) == 1:
                    hit += 1.0
                elif self.get_radar_level(self.real[0][i][j], thd_l, thd_r) == 1 and \
                        self.get_radar_level(self.predict[num][i][j], thd_l, thd_r) != 1:
                    miss += 1.0
                elif self.get_radar_level(self.real[0][i][j], thd_l, thd_r) != 1 and \
                        self.get_radar_level(self.predict[num][i][j], thd_l, thd_r) == 1:
                    fake += 1.0
                else:
                    white += 1.0
        return hit, miss, fake, white

    def border(self, x, y, ind, thd_l, thd_r):
        assert x >= 0 and x <= 799
        assert y >= 0 and y <= 799
        x0, x1, y0, y1 = 0, 0, 0, 0
        if x - 2 < 0:
            x0 = 0
        elif x + 2 > 799:
            x1 = 799
        else:
            x0 = x - 2
            x1 = x + 2
        if y - 2 < 0:
            y0 = 0
        elif y + 2 > 799:
            y1 = 799
        else:
            y0 = y - 2
            y1 = y + 2
        while y0 <= y1:
            while x0 <= x1:
                if self.get_radar_level(self.predict[ind][x0][y0], thd_l, thd_r) == 1:
                    return 1
                    break
                else:
                    x0 += 1
            y0 += 1
        return 0

    def stas_5_plus_5(self, num, level_ind):
        assert self.real[0].shape[0] == self.predict[num].shape[0]
        assert self.real[0].shape[1] == self.predict[num].shape[1]
        hit, miss, fake, white = 0.0, 0.0, 0.0, 0.0

        if level_ind == len(self.threshold) - 1:
            thd_l = self.threshold[level_ind]
            thd_r = float("inf")
        else:
            thd_l = self.threshold[level_ind]
            thd_r = self.threshold[level_ind + 1]

        for i in range(self.startX, self.endX):
            for j in range(self.startY, self.endY):
                if self.get_radar_level(self.real[0][i][j], thd_l, thd_r) == 1 and \
                        self.border(i, j, num, thd_l, thd_r) == 1:
                    hit += 1.0
                elif self.get_radar_level(self.real[0][i][j], thd_l, thd_r) == 1 and \
                        self.border(i, j, num, thd_l, thd_r) != 1:
                    miss += 1.0
                elif self.get_radar_level(self.real[0][i][j], thd_l, thd_r) != 1 and \
                        self.get_radar_level(self.predict[num][i][j], thd_l, thd_r) == 1:
                    fake += 1.0
                else:
                    white += 1.0
        return hit, miss, fake, white

    def calc_index(self, id):
        assert self.real[0].shape[0] == self.predict[id].shape[0]
        assert self.real[0].shape[1] == self.predict[id].shape[1]
        dict_index = {}
        for n in range(len(self.threshold)):
            if self.method == '1x1':
                hit, miss, fake, white = self.stas_1_plus_1(
                    num=id, level_ind=n)
            elif self.method == '5x5':
                hit, miss, fake, white = self.stas_5_plus_5(
                    num=id, level_ind=n)
            else:
                print(' Select Method ! ')
                break
            try:
                POD = hit / (hit + miss)
            except ZeroDivisionError:
                POD = 0
            try:
                FAR = fake / (hit + fake)
            except ZeroDivisionError:
                FAR = 0
            try:
                CSI = hit / (hit + miss + fake)
            except ZeroDivisionError:
                CSI = 0
            dict_index['POD_' + str(n)] = POD
            dict_index['FAR_' + str(n)] = FAR
            dict_index['CSI_' + str(n)] = CSI
        df = pd.DataFrame(dict_index, index=[0])

        return df

    def merge(self):
        assert len(self.real) == 1 and len(
            self.predict) == base_config['output_seq_length']

        if len(self.real) != 1 or len(self.predict) != base_config['output_seq_length']:
            print(" Incorrect Data Dims !!! ")
            sys.exit(0)
        else:
            df_list = []
            for x in range(len(self.predict)):
                df1 = self.calc_index(id=x)
                df_list.append(df1)
            df = pd.concat(df_list)
            df = df.reset_index()
            df = df.drop(columns='index')
            print(df)
            if (self.endX - self.startX) < self.real[0].shape[0] or \
                    (self.endY - self.startY) < self.real[0].shape[1]:
                df.to_csv(os.path.join(self.savepath,
                                       self.time + "_" + self.method + '_region_evaluate.csv'))
            else:
                df.to_csv(os.path.join(self.savepath,
                                       self.time + "_" + self.method + '_all_evaluate.csv'))
