
import os
import numpy as np
import math

from keras.preprocessing.image import load_img, img_to_array
#from keras.applications import vgg16
from keras.layers import Dense, Activation, Merge, Flatten, MaxPooling2D, Reshape, Masking
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils.np_utils import to_categorical
from keras.metrics import *
from keras.applications import vgg16
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras import initializers

import argparse

from rain.utils import *
from rain.sraettings import *
from scipy.optimize import least_squares

#from theano import tensor as TT
#import theano

import thread
thread.daemon = False

#theano.config.openmp = True
#theano.config.optimizer = "fast_compile"
#theano.config.traceback.limit=100

patch = (11, 9)


def fit(radar_maps, rain_maps, learning_rate=0.00001, max_steps=1000, epsilon=0.000001):
    A = 200.0
    b = 1.6
    last_loss = 0.0
    radar_maps = radar_maps*14.0*5
    for i in range(max_steps):
        for j, radar_map in enumerate(radar_maps):
            diff = np.zeros((2))
            for x in range(radar_map.shape[0]):
                for y in range(radar_map.shape[1]):
                    if rain_maps[j][x][y] > 0 and radar_maps[j][x][y] > 0:
                        den = 2*(A*rain_maps[j][x][y]**b-radar_maps[j][x][y])*rain_maps[j][x][y]**b

                        diff[0] += den
                        diff[1] += den*A*math.log(rain_maps[j][x][y])
                        print(i, j, x, y, A, b, radar_maps[j][x][y], rain_maps[j][x][y], A*rain_maps[j][x][y]**b, diff)
                        #diff = ().sum()

            A -= learning_rate * diff[0] * A
            b -= learning_rate * diff[1] * b

        loss = 0.0
        for j, radar_map in enumerate(radar_maps):
            loss += (((A*radar_maps[j]**b) - rain_maps[j])**2).sum()
        #print(i, A, b, loss, last_loss)

        if abs(last_loss - loss) < epsilon:
            break
        last_loss = loss
    return A, b

def radar_to_rain(radar_map, radar_map_1, A=300.0, b=1.4):
    Z0 = 10**(radar_map*7)
    Z1 = 10**(radar_map_1*7)
    rain_map = np.zeros(radar_map.shape)
    for x in range(radar_map.shape[0]):
        for y in range(radar_map.shape[1]):
            rain_map[x][y] = ((Z0[x][y]/A)**(1/b) + (Z1[x][y]/A)**(1/b))/2.0

    #plot_map(rain_map, colorbar=True, src='global_rain')
    return rain_map

def predict(radar_maps, rain_maps, A=300.0, b=1.6):
    for i, radar_map in enumerate(radar_maps[:2]):
        radar_to_rain(radar_maps[i], radar_maps[i-1])

def evaluate(Y_predict, Y_true, mask, th=10.0):
    hit, miss, fake, z = 0.0, 0.0, 0.0, 0.0
    for i, Y in enumerate(Y_true[1:]):
        for x in range(Y.shape[0]):
            for y in range(Y.shape[1]):
                if mask[x][y] == 1:
                    if Y_predict[i][x][y] >= th and Y_true[i][x][y] >= th:
                        hit += 1
                    elif Y_predict[i][x][y] >= th and Y_true[i][x][y] < th:
                        fake += 1
                    elif Y_predict[i][x][y] < th and Y_true[i][x][y] >= th:
                        miss += 1
                    else:
                        z += 1
                    pass
    print Y_predict.shape, Y_true.shape, Y_predict.max(), Y_true.max(), mask.shape
    print hit, miss, fake

    POD = hit/(hit+miss+1)
    FAR = fake/(hit+fake+1)
    CSI = hit/(hit+miss+fake+1)

    print POD, FAR, CSI

'''def quick_cost(prediction, target, cost_func, mask=None):
    assert prediction.ndim == target.ndim
    assert (mask.ndim == 2 and prediction.ndim == 5) if mask is not None else True
    print('Using cost func:', cost_func)
    if "SquaredLoss" == cost_func:
        if mask is None:
            return TT.sqr(prediction - target).sum()
        else:
            return (TT.sqr((prediction - target)) * TT.shape_padright(mask, 3)).sum()

    elif "BinaryCrossEntropy" == cost_func:
        if mask is None:
            return TT.nnet.binary_crossentropy(prediction, target).sum()
        else:
            return (TT.nnet.binary_crossentropy(prediction, target) * TT.shape_padright(mask, 3)).sum()
    elif "Fade" == cost_func:
        return TT.switch(TT.le(prediction, 0.1), 1.5*TT.sqr(prediction-target), TT.sqr(prediction-target)).sum()
    elif "Fade_rain" == cost_func:
        return TT.switch(TT.gt(prediction, target), 2.5*TT.sqr(prediction-target), TT.sqr(prediction-target)).sum()
    elif "TS" == cost_func:
        hit = TT.switch(TT.eq(get_rain_level(prediction), 0), 0, TT.switch(get_rain_level(prediction), get_rain_level(target), 1, 0)).sum()
        miss = TT.switch(TT.lt(get_rain_level(prediction), get_rain_level(target)), 1, 0).sum()
        fake = TT.switch(TT.gt(get_rain_level(prediction), get_rain_level(target)), 1, 0).sum()
        return hit/(hit+miss+fake)'''

def get_rain_level(vals):
    return TT.switch(TT.le(vals, 0.1), 0,
                     TT.switch(TT.le(vals, 2.5), 1,
                               TT.switch(TT.le(vals, 8.0), 2,
                                         TT.switch(TT.le(vals, 16.0), 3, 4))))

def TS_cost(y_true, y_pred):
    hit = TT.switch(TT.eq(get_rain_level(y_pred), 0), 0, TT.switch(get_rain_level(y_pred), get_rain_level(y_true), 1, 0)).sum()
    miss = TT.switch(TT.lt(get_rain_level(y_pred), get_rain_level(y_true)), 1, 0).sum()
    fake = TT.switch(TT.gt(get_rain_level(y_pred), get_rain_level(y_true)), 1, 0).sum()
    return hit/(hit+miss+fake)

def Square_cost(y_true, y_pred):
    return (TT.switch(TT.gt(y_true, 0),
                     TT.switch(TT.gt(y_pred, 0),
                               TT.sqr(y_pred-y_true), 0), 0)).sum()

def mask_cost(y_true, y_pred):
    mask = load_mask('../data/r1/201601/16010100.000')
    return K.mean(TT.switch(TT.gt(mask, 0), TT.sqr(y_pred-y_true), 0), axis=-1)

class RRLayer(Layer):
    def __init__(self, size=(110, 90), mask=None, patch=(11, 9),  **kwargs):
        self.size = size
        self.mask = mask
        self.patch = patch
        super(RRLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        patch_size = self.size[0]/self.patch[0]
        print('patch size', self.size, self.patch)
        self.A = self.add_weight(name='A', initializer=initializers.Constant(value=0.8),
                                #initializer=initializers.RandomNormal(mean=0.8, stddev=0.05, seed=None),
                                #initializer=initializers.RandomUniform(minval=0.75, maxval=0.85, seed=None),
                                #initializer = 'uniform',
                                #shape=(self.size[0]/10, self.size[1]/10),
                                shape=(self.patch[0]*self.patch[1]),
                                trainable=True)
        self.b = self.add_weight(name='B', initializer=initializers.Constant(value=1.4),
                                #initializer=initializers.RandomNormal(mean=1.4, stddev=0.05, seed=None),
                                #initializer=initializers.RandomUniform(minval=1.3, maxval=1.5, seed=None),
                                #initializer = 'uniform',
                                #shape=(self.size[0]/10, self.size[1]/10),
                                shape=(self.patch[0]*self.patch[1]),
                                trainable=True)

        super(RRLayer, self).build(input_shape)


    def call(self, x):
        patch_size = self.size[0]/self.patch[0]
        #x = K.reshape(x, (-1, self.size[0]/patch, self.size[1]/patch, patch*patch))
        #x = Flatten()(x)

        y = shifting(x, self.size[0]*self.size[1])
        y = patch_div(10**(y*7), self.A*300.0, patch_size)
        y = patch_exp(y, 1/self.b, patch_size)

        z = shifting(x, -self.size[0]*self.size[1])
        z = patch_div(10**(z*7), self.A*300.0, patch_size)
        z = patch_exp(z, 1/self.b, patch_size)


        x = patch_div(10**(x*7), self.A*300.0, patch_size)
        x = patch_exp(x, 1/self.b, patch_size)
        x = (3*x+2*y+z)/6

        #if self.mask is not None:
        #    x = x*self.mask

        #x = K.expand_dims(x, -1)
        #x = K.reshape(x, (-1, self.size[0], self.size[1], 1))
        #x = K.squeeze(x, -1)
        #x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1), ndim=4)
        return x

    def compute_output_shape(self, input_shape):
        #return input_shape[0]
        return (input_shape[0], self.size[0], self.size[1])


def rrmodel(size=(110, 90), mask=None, patch=(11, 9)):
    input_tensor = Input(shape=size)

    x = RRLayer(size=size, mask=mask, patch=patch, name='RRLayer')(input_tensor)


    #x = Reshape((size[0], size[1], 1))(input_tensor)
    #print('ndim1', K.ndim(x), x.shape, x._keras_shape, K.int_shape(x))


    #x = TT.expand_dims(x, -1)
    #x = Conv2D(128, (3, 3), padding='same', name='conv1')(x)
    #print('ndim2', K.ndim(x), x.shape, x._keras_shape)
    #x = Flatten()(x)
    #output = x
    #x = AveragePooling2D(2, name='pool')(x)
    
    #output = GlobalAveragePooling2D()(x)
    #x = Conv2DTranspose(1, 3, strides=1, padding='same', name='trans_conv')(x)
    #print('ndim3', K.ndim(x), x.shape, x._keras_shape)
    

    # element cost layer?

    #x = Reshape((size[0], size[1]))(x)
    #print('ndim4', K.ndim(x), x.shape, x._keras_shape)
    #output = Flatten()(x)
    #output = Masking(mask_value=0)(x)


    rms = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.9)
    model = Model(inputs=input_tensor, outputs=x)
    #model.compile(optimizer=rms, loss='mean_squared_error')
    model.compile(optimizer=rms, loss=mask_cost)

    return model

def train():
    mask = load_mask('../data/r1/201601/16010100.000')
    patch = (11, 9)
    model = rrmodel(size=(110, 90), mask=mask, patch=patch)


    print('mask:', mask.shape)
    #plt.pcolormesh(mask)
    #plt.show()
    trainX, trainY, _ = load(start_date, end_date, normalize=True)
    validX, validY, _ = load(valid_start_date, valid_end_date, normalize=True)
    testX, testY, test_dates = load(test_start_date, test_end_date, normalize=True)

    '''trainX = np.expand_dims(trainX, -1)
    validX = np.expand_dims(validX, -1)
    testX = np.expand_dims(testX, -1)
    rainY = np.expand_dims(trainY, -1)
    validY = np.expand_dims(validY, -1)
    testY = np.expand_dims(testY, -1)
    mask = np.expand_dims(mask, -1)'''
    print('load data', start_date, end_date, valid_start_date, valid_end_date, test_start_date, test_end_date)
    print('load data:', trainX.shape, trainY.shape, validX.shape, validY.shape, trainY.max(), validY.max())


    #radar_to_rain(trainX, trainY)
    #predict(trainX, trainY)

    #trainX = [[[1,2],[3, 4]], [[4,5], [5, 6]]]
    #trainY = np.reshape(trainY, (len(trainY), -1))
    #validY = np.reshape(validY, (len(validY), -1))
    #print('train data reshape:', trainY.shape, validY.shape)

    model.fit(trainX, trainY, epochs=10, batch_size=128, shuffle=True, validation_data=(validX, validY))
    model.save('model/rain_model.h5')
    '''for layer in model.layers:
        if layer.name == 'RRLayer':
            #print(layer.A, layer.b)
            print layer.get_weights()'''
    #get_weights = theano.function([model.layers[0].input], model.layers[1].output(train=False), allow_input_downcast=True)
    #weights = get_weights(trainX) # same result as above


    predictY = model.predict(testX, batch_size=64)
    print('prediction:', predictY.shape, predictY.max(), predictY.min(), testY.max())

    weights = model.get_layer('RRLayer').get_weights()
    print('weights', weights)
    #evaluate(predictY, testY, mask, th=2.)

    it = test_start_date
    

    for i in range(len(predictY)):
        result_dir = os.path.join(save_dir, it.strftime('%Y%m'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            os.makedirs(result_dir+'/rain')
        print(i, test_dates[i], predictY[i].shape, predictY[i].max(), testY[i].max())
        plot_map(predictY[i], src='global_rain', save=True, name=result_dir+'/rain/'+it.strftime('%Y%m%d%H')+'-modelmap.png', FJ=False, colorbar=True)
        it += datetime.timedelta(hours=1)
    


    '''model = rrmodel(size=200)
    print('Model training...')
    model.fit(trainX, trainY, nb_epoch=30, batch_size=32, shuffle=True, validation_data=(validX, validY))
    #model.fit(images, labels, nb_epoch=1, batch_size=64, shuffle=True, validation_split=0.2)
    

    predictY = model.predict(testX)
    for i, y in enumerate(predictY):
        print(y.shape, y.max(), trainY[0].max())
        if i == 0:
            plot_map(y*40.0, colorbar=True, src='global_rain')
            plot_map(trainY[0]*40.0, colorbar=True, src='global_rain')'''

def predict(start_date, end_date):
    model = load_model('model/rain_model.h5', custom_objects={'RRLayer': RRLayer, 'mask_cost': mask_cost})

    testX, testY, test_dates = load(start_date, end_date, normalize=True)
    predictY = model.predict(testX, batch_size=64)
    print('prediction:', predictY.shape, predictY.max(), predictY.min(), testY.max())

    it = start_date
    for i in range(len(predictY)):
        result_dir = os.path.join(save_dir, it.strftime('%Y%m'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            os.makedirs(result_dir+'/rain')
        print(i, test_dates[i], predictY[i].shape, predictY[i].max(), testY[i].max())
        plot_map(predictY[i], src='global_rain', save=True, name=result_dir+'/rain/'+it.strftime('%Y%m%d%H')+'-modelmap.png', FJ=False, colorbar=True)
        it += datetime.timedelta(hours=1)

def run():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating rain.')
    parser.add_argument('mode', metavar='base', type=str, default='run',
                        help='Mode: run, train')
    #parser.add_argument('--src', type=str, default='global_radar', required=False,
    #                    help='Type of data: global_radar, pgm, rain')


    args = parser.parse_args()
    mode = args.mode
    #src = args.src


    if mode == 'train':
        train()
    elif mode == 'predict':
        predict(test_start_date, test_end_date)
    elif mode == 'run':
        run()
