# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date: 2019-04-08

import xml
import xml.etree.ElementTree as ET
from PyQt5.QtCore import qCompress, qUncompress
from PIL import Image
import numpy as np
import os
import sys


def parse_xml(xml_data):
    tree = ET.fromstring(xml_data)
    print(tree.tag)
    print(tree.attrib)
    for childs in tree:
        if childs:
            print(childs.tag)
            print(childs.attrib)
        else:
            continue


def split_file(filepath):
    blob_dict = {}
    with open(filepath, 'rb') as dbz:
        total_length = len(dbz.read())
        # print(total_length)
        dbz.seek(0)
        while True:
            line = dbz.readline()
            if line:
                if 'END XML' in str(line):
                    length = dbz.tell()
                    dbz.seek(0)
                    xml_part = dbz.read(length)
                elif 'compression' in str(line):
                    # tag = str(line, encoding='utf-8')
                    tag = str(line)
                    tag_len = len(tag)
                    read_length = int(tag.split(" ")[2][6:-1])
                    data = dbz.read(read_length)
                    up_data = qUncompress(data)
                    blob_dict['blobid_' + tag.split(" ")[1][8:-1]] = up_data
                else:
                    continue
            else:
                break
    return xml_part, blob_dict


def parse_data(xml_data, dict_data, map1):
    info = parse_xml(xml_data)
    print("-----------------------------")
    data = dict_data['blobid_0']
    data_list = []
    print(len(data))
    if len(data) == 360000:
        trans_data = [ord(data[i]) for i in range(360000)]
    else:
        print(str(len(data)) + " < 36w")
        sys.exit()
    for i in range(len(trans_data)):
        if trans_data[i] != 0:
            trans_data[i] = map1[trans_data[i]]

    data_array = np.array(trans_data, dtype=np.int32)
    data_array = data_array.reshape((600, 600))
    data_array = np.flipud(data_array)

    # print(data_array)
    print(data_array.shape[0])
    return data_array


def plot(data, filepath, savepath):
    fig = Image.new(mode='RGB', size=(600, 600), color='white')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] < 0:
                fig.putpixel((i, j), (0, 172, 164))
                # 1111
            elif data[i][j] == 0:
                fig.putpixel((i, j), (255, 255, 255))
            elif data[i][j] <= 5 and data[i][j] > 0:
                fig.putpixel((i, j), (192, 192, 254))
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
    fig = fig.rotate(90)
    # fig = fig.transpose(Image.FLIP_LEFT_RIGHT)
    fig.save(os.path.join(savepath, name), 'PNG')


def checkdir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)


def main():
    filefolder = "/root/my/test/"
    savepath = "/root/my/testpng/"
    checkdir(savepath)
    dict1 = {i: j / 2 for i, j in zip(range(1, 255, 1), range(-63, 191, 1))}
    for filename in os.listdir(filefolder):
        filepath = os.path.join(filefolder, filename)
        xml_part, blob_info = split_file(filepath)
        raw_data = parse_data(
            xml_data=xml_part, dict_data=blob_info, map1=dict1)
        plot(raw_data, filepath, savepath)


if __name__ == '__main__':
    main()
