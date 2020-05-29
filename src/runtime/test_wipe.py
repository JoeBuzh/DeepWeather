# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date: 2019-06-12

'''
   The tool for checking noise-wipe
'''

import numpy as np
import json
import os
from PIL import Image
from parse_max import split_file, parse_data, parse_xml, checkdir


def plot(data, filepath, savepath, rows, columns):
    fig = Image.new(mode='RGBA', size=(rows, columns),
                    color=(255, 255, 255, 0))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] <= 5:
                fig.putpixel((i, j), (255, 255, 255, 0))
                # 2222
            elif data[i][j] <= 10 and data[i][j] > 5:
                fig.putpixel((i, j), (122, 114, 238))
                # 3333
            elif data[i][j] <= 15 and data[i][j] > 10:
                fig.putpixel((i, j), (30, 38, 208))
                # 4444
            elif data[i][j] <= 20 and data[i][j] > 15:
                fig.putpixel((i, j), (166, 252, 168))
                # 5555
            elif data[i][j] <= 25 and data[i][j] > 20:
                fig.putpixel((i, j), (0, 234, 0))
                # 6666
            elif data[i][j] <= 30 and data[i][j] > 25:
                fig.putpixel((i, j), (16, 146, 26))
                # 7777
            elif data[i][j] <= 35 and data[i][j] > 30:
                fig.putpixel((i, j), (252, 244, 100))
                # 8888
            elif data[i][j] <= 40 and data[i][j] > 35:
                fig.putpixel((i, j), (200, 200, 2))
                # 9999
            elif data[i][j] <= 45 and data[i][j] > 40:
                fig.putpixel((i, j), (144, 144, 0))
                # 1010
            elif data[i][j] <= 50 and data[i][j] > 45:
                fig.putpixel((i, j), (254, 172, 172))
                # 11-11
            elif data[i][j] <= 55 and data[i][j] > 50:
                fig.putpixel((i, j), (254, 100, 92))
                # 1212
            elif data[i][j] <= 60 and data[i][j] > 55:
                fig.putpixel((i, j), (238, 2, 48))
                # 1313
            elif data[i][j] <= 65 and data[i][j] > 60:
                fig.putpixel((i, j), (212, 142, 254))
                # 1414
            elif data[i][j] > 65:
                fig.putpixel((i, j), (170, 36, 250))
                # 1515
            else:
                continue

    name = filepath.split("/")[-1][:-7] + "_max_top.PNG"
    print(name)
    fig.save(os.path.join(savepath, name), 'PNG', mode='RGBA')


def read_noise():
    base_dir = os.getcwd()
    noise = os.path.join(base_dir, 'noise.json')
    if os.path.exists(noise):
        with open(noise, 'r') as f:
            data = json.load(f)
            noise_data = data['noise_pos']

    return noise_data


def read_max(filepath, dict1, noise):
    assert os.path.exists(filepath)
    if os.path.exists(filepath):
        print("0")
    xml_part, blob_info = split_file(filepath)
    rows, columns = parse_xml(xml_part)
    print(rows, columns)
    raw_data = parse_data(dict_data=blob_info, rows=rows,
                          columns=columns, map1=dict1)

    savepath = r'C:/Users/BZH/Desktop/test/'
    print(raw_data.shape)
    checkdir(savepath)
    print(filepath.split("/")[-1][:-7] + '_max_top.PNG')
    # plot(data=raw_data, filepath=filepath,
    #      savepath=savepath, rows=rows, columns=columns)

    wiped_data = wipe_noise(raw_data, noise)
    plot(data=wiped_data, filepath=filepath,
         savepath=savepath, rows=rows, columns=columns)


def wipe_noise(data, noise):
    for pos in noise:
        # data[data[pos[0]][pos[1]] < 5] = 0
        # print(data[pos[0]][pos[1]])
        # print(pos[-1])
        # print("=========")
        if abs(data[pos[0]][pos[1]] - pos[-1]) < 13:
            data[pos[0]][pos[1]] = 0

    return data


def main():
    base = r'D:/InsightValue/max/日实况/20180630实况数据/max_20180630/'
    # max_file = r'D:/InsightValue/max/日实况/20180630实况数据/max_20180630/201806301728dbz.max'
    dict1 = {i: j / 2 for i, j in zip(range(1, 255, 1), range(-63, 191, 1))}
    noise = read_noise()
    # print(noise[0])
    # print(noise[0][1])
    for file in os.listdir(base):
        if file
        max_file = os.path.join(base, file)
        read_max(max_file, dict1, noise)


if __name__ == '__main__':
    main()
