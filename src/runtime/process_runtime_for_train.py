# -*- coding: utf-8 -*-
# date: 2019-2-25
# Author: Joe-BU

import datetime
import imghdr
import os
import shutil
from PIL import Image
import numpy as np
import sys
from settings import *
from mylog import mylog as lg

preprocess_log = lg.init_logger(preprocess_log_path)


def crop(input_dir, filename, output_dir):
    img = Image.open(os.path.join(input_dir, filename))
    img = img.convert('RGB')
    q = np.asarray(img)
    q = q[0:800, 224:1024]
    q = q[100:500, 300:700]
    final = Image.fromarray(q.astype('uint8')).convert('RGB')
    final.save(os.path.join(output_dir, filename))


def wipeBaseMap(local_path, filename):
    img = Image.open(os.path.join(local_path, filename))
    width = img.size[0]
    height = img.size[1]
    for i in range(0, width):
        for j in range(0, height):
            data = (img.getpixel((i, j)))
            if (data[0] == 1 and data[1] == 160 and data[2] == 246):
                n = 1
            elif (data[0] == 0 and data[1] == 236 and data[2] == 236):
                n = 1
            elif (data[0] == 0 and data[1] == 216 and data[2] == 0):
                n = 1
            elif (data[0] == 1 and data[1] == 144 and data[2] == 0):
                n = 1
            elif (data[0] == 255 and data[1] == 255 and data[2] == 0):
                n = 1
            elif (data[0] == 231 and data[1] == 192 and data[2] == 0):
                n = 1
            elif (data[0] == 255 and data[1] == 144 and data[2] == 0):
                n = 1
            elif (data[0] == 255 and data[1] == 0 and data[2] == 0):
                n = 1
            elif (data[0] == 214 and data[1] == 0 and data[2] == 0):
                n = 1
            elif (data[0] == 192 and data[1] == 0 and data[2] == 0):
                n = 1
            elif (data[0] == 255 and data[1] == 0 and data[2] == 240):
                n = 1
            elif (data[0] == 150 and data[1] == 0 and data[2] == 180):
                n = 1
            elif (data[0] == 173 and data[1] == 144 and data[2] == 240):
                n = 1
            else:
                img.putpixel((i, j), (255, 255, 255))
                n = 1
    img = img.convert("RGB")
    img.save(os.path.join(local_path, filename))


def format_img(path1, filename, path2):
    name_new = filename[:12] + '.PNG'  # local env
    # name_new =
    crop(path1, filename, path2)
    os.rename(os.path.join(path2, filename), os.path.join(path2, name_new))
    wipeBaseMap(local_path=path2, filename=name_new)


def check_dir(path0):
    if not os.path.exists(path0):
        os.makedirs(path0)
    assert os.path.exists(path0)


def main():
    raw_path = raw_image_path + '201904/'
    to_fold = train_pool
    check_dir(raw_path)
    check_dir(to_fold)

    time_now = datetime.now() - timedelta(hours=6)
    time_start = time_now - timedelta(days=1)
    while (time_start <= time_now):
        target = time_start.strftime("%Y%m%d%H%M") + '.PNG'
        if os.path.exists(os.path.join(raw_path, target)) and imghdr.what(os.path.join(raw_path, target)) != None:
            format_img(path1=raw_path, filename=target, path2=to_fold)
        else:
            preprocess_log.info(target + " doesn't exist or was broken !!!")
            pass
        time_start += timedelta(minutes=6)
    preprocess_log.info(time_now.strftime("%Y-%m-%d %H:%M") + " preprocess train data done !")
    preprocess_log.info("-------  -------  -------  ------  ------  -------\r")


if __name__ == '__main__':
    print datetime.now()
    main()
    print datetime.now()
