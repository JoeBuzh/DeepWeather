# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date: 2019-04-16

import os
from datetime import datetime, timedelta
import shutil


def process(dir0, srt, end, dir1):
    assert srt <= end
    for filename in os.listdir(dir0):
        filetime = datetime.strptime(filename[:12], "%Y%m%d%H%M")
        if filetime >= srt and filetime <= end:
            try:
                checkdir(os.path.join(dir1, filetime.strftime("%Y%m")))
                if os.path.exists(os.path.join(dir1, filetime.strftime("%Y%m"),
                                               filetime.strftime("%Y%m%d%H%M") + 'dBZ.max')):
                    continue
                else:
                    shutil.copy(os.path.join(dir0, filename), os.path.join(
                        os.path.join(dir1, filetime.strftime("%Y%m")), filename))
            except IOError as e:
                print(str(e))
                break
            else:
                name_new = filename[:12] + "dBZ.max"
                os.rename(os.path.join(os.path.join(dir1, filetime.strftime("%Y%m")), filename),
                          os.path.join(os.path.join(dir1, filetime.strftime("%Y%m")), name_new))
        else:
            continue


def checkdir(path1):
    if not os.path.exists(path1):
        os.makedirs(path1)
    assert os.path.exists(path1)


def remove(rawdir):
    for filename in os.listdir(rawdir):
        if filename[-3:] == 'tmb':
            # print(filename)
            os.remove(os.path.join(rawdir, filename))


def main():
    raw_dir = r'D:/InsightValue/max/2017/201709/'
    dst_dir = r'D:/InsightValue/max/2017/201709_rename/'
    start = datetime(2017, 9, 1, 0, 0)
    interval = 5
    end = datetime(2017, 9, 30, 23, 55)
    checkdir(dst_dir)
    process(raw_dir, start, end, dst_dir)


if __name__ == '__main__':
    main()
