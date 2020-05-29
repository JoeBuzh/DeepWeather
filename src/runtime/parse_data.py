# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date: 2019-04-08

import xml
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import qCompress, qUncompress
from PIL import Image
import numpy as np
import os
import sys
import json


def parse_xml(xml_data):
    tree = ET.fromstring(xml_data)
    element_data = [i for i in tree if i.tag == 'data'][0]
    picture_info = [m for m in element_data if m.tag ==
                    'radarpicture' and m.attrib['placeid'] == 'top'][0]
    rows = [n for n in picture_info if n.tag ==
            'datamap' and n.attrib['blobid'] == "0"][0].attrib['rows']
    columns = [n for n in picture_info if n.tag ==
               'datamap' and n.attrib['blobid'] == "0"][0].attrib['columns']
    return int(rows), int(columns)


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


def parse_data(dict_data, rows, columns, map1):
    data = dict_data['blobid_0']
    data_list = []
    if len(data) == rows * columns:
        trans_data = [ord(data[i]) for i in range(rows * columns)]
    else:
        print(str(len(data)) + " < " + str(rows * columns))
        sys.exit()
    trans_data = [map1[trans_data[i]] if trans_data[i] != 0 else -999
                  for i in range(len(trans_data))]
    data_array = np.array(trans_data, dtype=np.dtype("float32"))
    # data_array = data_array.reshape((rows, columns))
    data_array = np.transpose(data_array.reshape((rows, columns)))

    return data_array   # Todo: paese more info


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

    # name = filepath.split("/")[-1][:-11] + "_max_top.PNG"
    fig.save(os.path.join(savepath, filepath), 'PNG', mode='RGBA')


def checkdir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)

def transparent(filename):
    assert os.path.exists(filename)
    img = Image.open(filename)
    img = img.convert('RGBA')
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            data = (img.getpixel((i,j)))
            if (data[0]==255 and data[1]==255 and data[2]==255):
                img.putpixel((i,j), (255, 255, 255, 0))
            else:
                continue
    img.save(filename, 'png', mode='RGBA')

def main():
    # test
    filefolder = "C:/Users/BZH/Desktop/201606/"
    pngpath = "C:/Users/BZH/Desktop/testpng/"
    jsonpath = "C:/Users/BZH/Desktop/testjson/"
    checkdir(pngpath)
    checkdir(jsonpath)
    dict1 = {i: j / 2 for i, j in zip(range(1, 255, 1), range(-63, 191, 1))}
    for filename in os.listdir(filefolder):
        filepath = os.path.join(filefolder, filename)
        xml_part, blob_info = split_file(filepath)
        rows, columns = parse_xml(xml_part)
        print(rows, columns)
        raw_data = parse_data(dict_data=blob_info,
                              rows=rows, columns=columns, map1=dict1)
        header = {"lat_lr": 38.6608, "lat_ul": 41.358,
                  "lon_lr": 118.452, "lon_ul": 114.927,
                  "dx": (41.358 - 38.6608) / rows,
                  "dy": (118.452 - 114.927) / columns,
                  "ny": columns,
                  "nx": rows}
        record = [{"data": raw_data.tolist()}, {"header": header}]
        with open(os.path.join(jsonpath, filename[:-11] + '.json'), 'w') as jsonfile:
            json.dump(record, jsonfile)
        plot(raw_data, filepath, pngpath, rows=rows, columns=columns)


if __name__ == '__main__':
    main()
