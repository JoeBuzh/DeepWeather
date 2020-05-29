
import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class CombineLayer(Layer):
    def __init__(self, layer_param):
        super(CombineLayer, self).__init__(layer_param)
        assert self.input.ndim == 5 or self.input.ndim == 4
        self.input_padding = layer_param.get('input_padding', None)
        self.conv_type = layer_param.get('conv_type', "valid")
        self.input_receptive_field = layer_param['input_receptive_field']
        self.input_stride = layer_param['input_stride']
        self.activation = layer_param['activation']


        self.combine_input = layer_param['combine_input']

        self.feature_out /= 2

        self.kernel_size = (
            self.feature_out, self.feature_in, self.input_receptive_field[0], self.input_receptive_field[1])
        self.W_xo = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xo"))
        self.b_o = quick_zero((self.feature_out, ), self._s("b_o"))

        self.param = [self.W_xo, self.b_o]
        self.fprop()

    def set_name(self):
        self.name = "CombineLayer-" + str(self.id)

    def step_fprop(self, input):
        print('Combine layer ', self.conv_type, input.ndim, input.shape, self.dim_out, self.kernel_size)
        # input = (3, 1, 128, 25, 25), reshape_input=(3, 128, 25, 25), kernel_size = (16, 128, 1, 1)
        reshape_input = input.reshape((input.shape[0] * input.shape[1], input.shape[2],
                                       input.shape[3], input.shape[4]))
        # self.W_xo = (16, 128, 1, 1) output = (6, 16, 25, 25)
        output = conv2d_same(input=reshape_input, filters=self.W_xo, input_shape=(None,) + self.dim_in,
                             filter_shape=self.kernel_size) + self.b_o.dimshuffle('x', 0, 'x', 'x')
        output = output.reshape((self.input.shape[0], input.shape[1]) + self.dim_out)
        output = quick_activation(output, self.activation)

        return output

    def fprop(self):
        input = TT.concatenate([self.input, self.combine_input])
        #input = self.input
        self.output = self.step_fprop(input)
