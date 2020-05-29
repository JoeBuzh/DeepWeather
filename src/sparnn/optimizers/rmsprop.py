import numpy
import time
import theano
import logging
import theano.tensor as TT
from sparnn.utils import *
from sparnn.optimizers import Optimizer
logger = logging.getLogger(__name__)
from model_config import base_config
from mylog import mylog as lg

# opt_log = lg.init_logger(base_config['deduce_log_path'])


class RMSProp(Optimizer):
    """
        RMSProp Introduced by Hinton(http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    """

    def __init__(self,
                 model,
                 train_data_iterator0,
                 train_data_iterator1,
                 train_data_iterator2,
                 train_data_iterator3,
                 train_data_iterator4,
                 train_data_iterator5,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param,
                 wind_train_iterator=None, wind_valid_iterator=None, wind_test_iterator=None
                 ):
        super(RMSProp, self).__init__(model,
                                      train_data_iterator0, train_data_iterator1,
                                      train_data_iterator2, train_data_iterator3,
                                      train_data_iterator4, train_data_iterator5,
                                      valid_data_iterator, test_data_iterator, hyper_param,
                                      wind_train_iterator=wind_train_iterator,
                                      wind_valid_iterator=wind_valid_iterator,
                                      wind_test_iterator=wind_test_iterator)
        self.learning_rate = numpy_floatX(hyper_param["learning_rate"])
        self.decay_rate = numpy_floatX(hyper_param["decay_rate"])

    def set_name(self):
        self.name = "RMSProp-" + self.id

    def get_update_func(self):
        print('*** Update Function of Rmsprop ......')
        # opt_log.info("*** Update Function of Rmsprop ......")
        updates = []
        lr = TT.scalar(self._s("learning_rate"), dtype=theano.config.floatX)
        rho = TT.scalar(self._s("decay_rate"), dtype=theano.config.floatX)
        eps = numpy_floatX(1E-6)
        self.meansquare = [theano.shared(p.get_value(
        ) * numpy_floatX(0.), name="%s.meansquare" % p.name) for p in self.model.param]
        g_msnew_list = [rho * g_ms + (1 - rho) * (TT.square(g))
                        for g, g_ms in zip(self.grad, self.meansquare)]
        updates += [(g_ms, g_msnew)
                    for g_ms, g_msnew in zip(self.meansquare, g_msnew_list)]
        updates += [(p, p - lr * g / TT.sqrt(g_msnew + eps))
                    for p, g, g_msnew in zip(self.model.param, self.grad, g_msnew_list)]
        return self.model.get_update_func(updates, [lr, rho])

    def learning_param(self):
        return [self.learning_rate, self.decay_rate]

    def print_stat(self):
        super(RMSProp, self).print_stat()
        logger.info("   Learning Parameters:")
        logger.info("      Clipping Threshold: " + str(self.clip_threshold))
        logger.info("      Learning Rate: " + str(self.decay_rate))
        logger.info("      Decay Rate: " + str(self.decay_rate))
