# -*- coding: utf-8 -*-

import pickle
import pprint
import factorWeather
import pygrib
import datetime
import sys   
import factorWeather


sys.setrecursionlimit(1000000) #例如这里设置为一百万  
model1 = factorWeather()
dict_config = {}
with open('/root/my/weather_online/model/DeepWeatherModel-test-validation-best.pkl', 'rb') as f:
    model1 = pickle.load(f)
    # model2 = model1.to_dict()
    # print(type(model1))
    # print('The model confioguration:')
    # print('\n'.join(['%s:%s' % item for item in model1.__dict__.items()]))
    for item in model1.__dict__.items():    
        dict_config[item[0]] = item[1]
    # print("-------------------------------------------------------")
#print('para dict:')
#print(dict_config) 
#print("------------------------------------------------------")

dict_config['config'] = {'size': (200, 200), 'data_dir': '/root/my/weather_online/save/first', 'max_epoch': 20, 'output_seq_length': 10, 'minibatch_size': 15, 'patch_size': 2, 'start_date': datetime.datetime(2018, 7, 3, 17, 0), 'input_data_type': 'float32', 'end_date': datetime.datetime(2018, 8, 18, 0, 0), 'input_seq_length': 5, 'kernel_num': (64, 64, 64), 'max': 14.0, 'learning_rate': 0.003, 'layer_num': 6, 'compress': 2, 'use_input_mask': False, 'layer_name': 'lstm', 'offset': 0.0, 'name': 'radar', 'level': 0, 'interval': 6, 'cost_func': 'Fade', 'cmap': 'radar', 'model_path': '/data/huangqiuping/weather/model', 'is_output_sequence': True, 'save_dir': '/root/my/weather_online/save/radar', 'kernel_size': 3}

#print('modify dict:')
#print(dict_config)
#print("-------------------------------------------------------")
model_modify = file('model_5in_10out.pkl','wb')
pickle.dump(dict_config,model_modify)
model_modify.close()
