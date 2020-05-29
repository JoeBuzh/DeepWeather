# -*- coding: utf-8 -*-
# date: 2018-12-27 21:19
# Author: Joe-BU

import os
import shutil
import sys
from datetime import datetime, timedelta
from settings import *
from mylog import mylog as lg

remove_log = lg.init_logger(remove_log_path)


# def calc_size():
#     pass


def remove_all(path1):
    try:
        shutil.rmtree(path1)
        remove_log.info(" Clean train data done !")
    except Exception as e:
        remove_log.error(" Clean error ", e)


def remove_folder(path1):
    for dirpath, dirnames, filenames in os.walk(path1):
        for filename in filenames:
            if os.path.isfile(os.path.join(dirpath, filename)):
                os.remove(os.path.join(dirpath, filename))
        shutil.rmtree(dirpath)


def remove_part(path1, t):
    for dir in os.listdir(path1):
        if os.path.isfile(os.path.join(path1, dir)):
            os.remove(os.path.join(path1, dir))
        elif datetime.strptime(dir, "%Y%m%d%H%M") + timedelta(hours=7) < datetime.strptime(t, "%Y%m%d%H%M"):
            try:
                remove_folder(os.path.join(path1, dir)) 
                remove_log.info(" Clean prediction folder [" + dir + "]")
            except Exception as e:
                #remove_log.error(" Clean prediction folder error : ", e)
                print str(e)
        else:
            pass


def main():
    check1 = to_path
    check2 = train_pool
    check3 = radar_save_dir
    check_time = datetime.now().strftime("%Y%m%d%H%M")
    remove_part(check1, check_time)
    remove_all(check2)
    remove_part(check3, check_time)


if __name__ == '__main__':
    main()
