# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date:2018-12-19

from model_config import base_config
import imghdr
import os
from datetime import datetime, timedelta
from PIL import Image
import sys
sys.path.append(base_config['append_path'])


def check_last(path0, time2):
    filename = time2.strftime("%Y%m%d%H%M") + '.PNG'
    if os.path.exists(os.path.join(path0, filename)) and imghdr.what(os.path.join(path0, filename)) != None:
        return filename
    else:
        for i in range(1, 100):
            filename1 = (time2 - timedelta(minutes=(6 * i))
                         ).strftime("%Y%m%d%H%M") + '.PNG'
            if os.path.exists(os.path.join(path0, filename1)) and imghdr.what(os.path.join(path0, filename1)) != None:
                break
        return filename1


def check_next(path0, time2):
    filename = time2.strftime("%Y%m%d%H%M") + '.PNG'
    if os.path.exists(os.path.join(path0, filename)) and imghdr.what(os.path.join(path0, filename)) != None:
        return filename
    else:
        for i in range(1, 100):
            filename1 = (time2 + timedelta(minutes=(6 * i))
                         ).strftime("%Y%m%d%H%M") + '.PNG'
            if os.path.exists(os.path.join(path0, filename1)) and imghdr.what(os.path.join(path0, filename1)) != None:
                break
        return filename1


def extract_last(path0, time1, path1):
    with Image.open(os.path.join(path0, check_last(path0, time1)), 'r') as f:
        f.save(os.path.join(path1, time1.strftime("%Y%m%d%H%M") + '.png'))


def extract_next(path0, time1, path1):
    with Image.open(os.path.join(path0, check_next(path0, time1))) as f:
        f.save(os.path.join(path1, time1.strftime("%Y%m%d%H%M") + '.png'))


def check_dir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)


def main():
    predict_moment = datetime.strptime(sys.argv[1], "%Y%m%d%H%M")
    raw_img_path = raw_image_path
    to_folder = to_path
    check_dir(to_folder)
    extract_to = os.path.join(to_folder, predict_moment.strftime("%Y%m%d%H%M"))
    check_dir(extract_to)
    check_dir(os.path.join(extract_to, 'input10'))
    check_dir(os.path.join(extract_to, 'true20'))

    for i in range(10):
        extract_last(path0=os.path.join(raw_img_path, predict_moment.strftime('%Y%m')),
                     time1=predict_moment - timedelta(minutes=(6 * i)),
                     path1=os.path.join(extract_to, 'input10'))

    for j in range(1, 21):
        extract_next(path0=os.path.join(raw_img_path, predict_moment.strftime("%Y%m")),
                     time1=predict_moment + timedelta(minutes=(6 * j)),
                     path1=os.path.join(extract_to, 'true20'))


if __name__ == '__main__':
    main()
