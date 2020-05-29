'''
author = zblhero@gmail.com
'''
import json

from flask import Flask
from flask import render_template
from flask import request, url_for

import pygrib

import sys
sys.path.append('/static')
sys.path.append('../src')
sys.path.append('./src')
sys.path.append('../')

from utils import *
from settings import *

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():
    return render_template('index.html')
    #return "Hello World!"

@app.route("/rest/latlons", methods=['GET', 'POST'])
def latlons():
    data = {}
    gr = pygrib.open('../tmp/ana.20170701/ana.201707010000.grib2')
    for g in gr:
        break

    lats, lons = g.latlons()
    lats = lats[startX:endX, startY:endY]
    lons = lons[startX:endX, startY:endY]

    data['lats'] = lats.tolist()
    data['lons'] = lons.tolist()

    return json.dumps(data)

@app.route("/rest/query", methods=['GET', 'POST'])
def query():
    name = request.args.get('name', '')
    level = request.args.get('level', '')
    time = request.args.get('time', datetime.datetime.now())
    time = datetime.datetime.strptime(time, '%Y%m%d%H0000')
    predict = int(request.args.get('predict', 0))
    #date = datetime.datetime.strptime(request.args.get('date'), '%Y%m%d%H0000')
    
    data = {}
    values = np.array([])
    print('isPredict', predict, name, level, time)

    result_dir = os.path.join(save_dir, time.strftime('%Y%m'))

    
    if predict != 0:  # predict data
        if name == 'Global Radar':
            pass
        else:
            result_name = it.strftime('%Y%m%d%H')+'00-'+name+'-'+level+'-'+str(predict)
            #values = read_txt(os.path.join(save_dir, '%s.%s.%d.txt'%(time.strftime('%Y%m%d%H'), name, level)) )
            values = read_json(os.path.join(result_dir, '%s.%s.%d.json'%(time.strftime('%Y%m%d%H'), name, level)))
    else:   # in situ data
        # get 200*200 values 
        if name == 'Global Radar':
            obj_text = codecs.open(os.path.join(data_dir, '201706010100.json'), 'r', encoding='utf-8').read()
            values = np.array(json.loads(obj_text))*5+10
        else:
            values = read_grib2(os.path.join(data_dir, 'ana.20170701/ana.201707010000.grib2'), name, level)
        print(values.shape) # 100 100 for global radar

    values = values.tolist()
    values = [[round(item, 2) for item in row] for row in values]
    data['values'] = values

    return json.dumps(data)

@app.route("/rest/prediction", methods=['GET', 'POST'])
def prediction():
    name = request.args.get('name', '')
    level = request.args.get('level', '')
    time = request.args.get('time', datetime.datetime.now())
    time = datetime.datetime.strptime(time, '%Y%m%d%H0000')
    row = int(request.args.get('row', 0))
    col = int(request.args.get('col', 0))

    
    result = {'name': name, 'value': [2, 4, 2, 3], 'prob': [0.7, 0.1, 0.6, 0.5, 0.02], 'alerts':[]}
    print('predict result:', result)
    return json.dumps(result)

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
