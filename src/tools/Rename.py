# -*- coding: utf-8 -*-
# Author: Joe-BU
# Datetime: 2018-12-18 15:04

from datetime import datetime, timedelta
import os
from settings import *

'''
QCCYGGggn.XXZ
Q：表示是雷达数据文件；
CC----编发该文件的站代码	CCCC的后两个字符
Y----日 Y=1为1;Y=2为2;Y=3为3;Y=4为4;Y=5为5;
        Y=6为6; Y=7为7; Y=8为8; Y=9为9; Y=A为10;
        Y=B为11; Y=C为12; Y=D为13; Y=E为14; Y=F为15;
        Y=G为16; Y=H为17; Y=I为18; Y=J为19; Y=K为20;
        Y=L为21; Y=M为22; Y=N为23; Y=O为24; Y=P为25;
        Y=Q为26; Y=R为27; Y=S为28; Y=T为29; Y=U为30;
        Y=V为31
GG ---小时
gg-----时间中的分钟
n-----一分钟内的顺序号
XX---雷达图象的资料要素及产品名
Z-----雷达图象的产品属性及距离
'''
# 2018 11 28 15 24.png
# 2018 11 28 15 24-{0,1,2}.png


def Format_Name(filename):
    Day_dict = {'01': '1', '02': '2', '03': '3', '04': '4', '05': '5', '06': '6',
                '07': '7', '08': '8', '09': '9', '10': 'A', '11': 'B', '12': 'C',
                '13': 'D', '14': 'E', '15': 'F', '16': 'G', '17': 'H', '18': 'I',
                '19': 'J', '20': 'K', '21': 'L', '22': 'M', '23': 'N', '24': 'O',
                '25': 'P', '26': 'Q', '27': 'R', '28': 'S', '29': 'T', '30': 'U',
                '31': 'V'}
    filename = 'QCC' + filename.replace(filename[:2], Day_dict[filename[:2]], 1)
    return filename


def Rename(predict_radar):
    for file in os.listdir(predict_radar):
        if file[12] == '.':
            time_now = datetime.strptime(file[:12], '%Y%m%d%H%M')
            src = os.path.join(predict_radar, file)
            img = time_now.strftime("%Y%m%d%H%M")[6:]
            img2 = Format_Name(filename=img)
            if file.endswith('.png'):
                dst = os.path.join(predict_radar, img2 + '.png')
            else:
                dst = os.path.join(predict_radar, img2 + '.json')
            try:
                os.rename(src, dst)
            except:
                continue
        if file[13] == '0':
            src = os.path.join(predict_radar, file)
            time_now = datetime.strptime(file[:12], '%Y%m%d%H%M')
            time_now_0 = (time_now + timedelta(minutes=6)
                          ).strftime("%Y%m%d%H%M")
            img = time_now_0[6:]
            img2 = Format_Name(filename=img)
            if file.endswith('.png'):
                dst = os.path.join(predict_radar, img2 + '.png')
            else:
                dst = os.path.join(predict_radar, img2 + '.json')
            try:
                os.rename(src, dst)
            except:
                continue
        if file[13] == '1':
            src = os.path.join(predict_radar, file)
            time_now = datetime.strptime(file[:12], '%Y%m%d%H%M')
            # print(time_now)
            time_now_1 = (time_now + timedelta(minutes=12)
                          ).strftime("%Y%m%d%H%M")
            # print(time_now_1)
            img = time_now_1[6:]
            # print(img)
            img2 = Format_Name(filename=img)
            # print(img2)
            if file.endswith('.png'):
                dst = os.path.join(predict_radar, img2 + '.png')
            else:
                dst = os.path.join(predict_radar, img2 + '.json')
            try:
                os.rename(src, dst)
            except:
                continue
        if file[13] == '2':
            # print(file)
            src = os.path.join(predict_radar, file)
            time_now = datetime.strptime(file[:12], '%Y%m%d%H%M')
            time_now_2 = (time_now + timedelta(minutes=18)
                          ).strftime("%Y%m%d%H%M")
            img = time_now_2[6:]
            img2 = Format_Name(filename=img)
            if file.endswith('.png'):
                dst = os.path.join(predict_radar, img2 + '.png')
            else:
                dst = os.path.join(predict_radar, img2 + '.json')
            try:
                os.rename(src, dst)
            except:
                continue


def main():
    # predict_path = radar_save_path
    predict_path = four_outputs
    Rename(predict_radar=predict_path)


if __name__ == '__main__':
    main()
