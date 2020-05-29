# -*- coding: utf-8 -*-
# Author: Joe BU
# Date: 2019-3-26 18:00:00

from model_config import base_config
import os
from datetime import datetime, timedelta
import sys
sys.path.append(base_config['append_path'])
from mylog import mylog as lg


runtime_log = lg.init_logger(base_config['runtime_log_path'])

def main(t1):
    #start_time = datetime(2018, 6, 30, 10, 0, 0)
    #end_time = datetime(2018, 6, 30, 11, 0, 0)
    start_time = t1
    end_time = start_time
    while start_time <= end_time:
        now = start_time.strftime("%Y-%m-%d %H:%M")
        print start_time
        runtime_log.info(' Predict time: ' + now)

        try:
            os.system('/data/anaconda2/bin/python /data/python_scripts/Pre_processor.py ' + 
                start_time.strftime("%Y%m%d%H%M"))
            runtime_log.info(' 1. Pre_precess Done!')
        except Exception as e:
            runtime_log.info(' Pre_preocess err: ' + e)
            sys.exit(0)

        try:
            os.system('/data/anaconda2/bin/python /data/python_scripts/bg_Transparent.py ' + 
                start_time.strftime("%Y%m%d%H%M"))
            runtime_log.info(' 2. Background Transparent!')
        except Exception as e:
            runtime_log.info(' Calc err: ' + e)
            sys.exit(0)

        try:
            os.system('/data/anaconda2/bin/python /data/weather_update/src/runtime/deduce_runtime.py predict --src radar '+
                start_time.strftime("%Y%m%d%H%M") + " " + start_time.strftime("%Y%m%d%H%M"))
            runtime_log.info(' 3.Predict ' + now + ' Done!')
        except Exception as e:
            runtime_log.info(' Predict err: ' + e)
            sys.exit(0)

        print "- > -> -> End " + now
        start_time = start_time + timedelta(minutes=6)


if __name__ == '__main__':
    runtime_log.info(' Start...')
    time_now = datetime.now()
    t = datetime(time_now.date().year, time_now.date().month, time_now.date().day, time_now.time().hour, time_now.time().minute, 0)
    main(t)
    runtime_log.info(' End !!!')
    runtime_log.info(' ========================================\n')
