import numpy
import time
import theano
import logging
import cPickle
import os
import os.path
import theano.tensor as TT
import numpy as np

#logger = logging.getLogger(__name__)
from sparnn.models import Model
from sparnn.utils import *
from model_config import base_config
import utils
from mylog import mylog as lg

'''
"autosave_mode": "best"

'''
train_log = lg.init_logger(base_config['train_log_path'])
process_log = lg.init_logger(base_config['process_log_path'])


class Optimizer(object):
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
                 wind_train_iterator=None, wind_valid_iterator=None, wind_test_iterator=None):
        self.model = model
        self.train_data_iterator0 = train_data_iterator0
        self.train_data_iterator1 = train_data_iterator1
        self.train_data_iterator2 = train_data_iterator2
        self.train_data_iterator3 = train_data_iterator3
        self.train_data_iterator4 = train_data_iterator4
        self.train_data_iterator5 = train_data_iterator5

        self.valid_data_iterator = valid_data_iterator
        self.test_data_iterator = test_data_iterator
        self.id = hyper_param["id"]
        self.start_epoch = hyper_param.get("start_epoch", 0)
        self.max_epoch = hyper_param["max_epoch"]
        self.autosave_mode = hyper_param.get("autosave_mode", None)
        self.do_shuffle = hyper_param.get("do_shuffle", None)
        self.save_path = hyper_param.get("save_path", "./")
        self.model_name = hyper_param.get("model_name", "model-proceed.pkl")
        self.save_interval = hyper_param.get("save_interval", None)
        self.max_epochs_no_best = hyper_param.get("max_epochs_no_best", None)
        self.clip_threshold = numpy_floatX(
            hyper_param['clip_threshold']) if 'clip_threshold' in hyper_param else None
        self.verbose = hyper_param.get("verbose", None)
        self.best_validation_error = numpy.inf
        self.current_validation_error = numpy.inf
        self.current_epoch = self.start_epoch

        self.validation_error_dict = {}

        self.wind_train_iterator = wind_train_iterator
        self.wind_valid_iterator = wind_valid_iterator
        self.wind_test_iterator = wind_test_iterator

        self.get_grad_param()
        if self.verbose:
            self.grad_norm_func = theano.function(
                self.model.interface_layer.symbols(), self.grad_norm)
        self.set_name()
        process_log.info("...Begin Building " +
                         self.name + " Updating Function...")
        train_log.info("...Begin Building " +
                       self.name + " Updating Function...")
        self.update_func = self.get_update_func()
        process_log.info("...Finished, Update Function Saved to " +
                         os.path.abspath(self.save_path))
        train_log.info("...Finished, Update Function Saved to ->->-> " +
                       os.path.abspath(self.save_path))

    def set_name(self):
        self.name = "Optimizer-" + self.id

    def get_grad_param(self):
        self.grad_norm = TT.sqrt(sum(TT.sqr(g).sum() for g in self.model.grad)) / TT.cast(
            self.model.interface_layer.input.shape[1], 'float32')
        # self.has_numeric_error = TT.or_(TT.isnan(self.grad_norm), TT.isinf(self.grad_norm))
        # self.grad = [TT.switch(self.has_numeric_error, numpy_floatX(0.1) * p, g)
        # for g, p in zip(self.model.grad, self.model.param)]
        self.grad = [g / TT.cast(
            self.model.interface_layer.input.shape[1], 'float32') for g in self.model.grad]
        if self.clip_threshold is not None:
            self.grad = [TT.switch(TT.ge(self.grad_norm, self.clip_threshold),
                                   g * self.clip_threshold / self.grad_norm, g) for g in self.grad]

    def get_update_func(self):
        process_log.info('*********update func in optimizer...')
        train_log.info(' ******** update func in optimizer .... ')
        return lambda x: x

    def learning_param(self):
        return None

    def combine_train(self):
        self.model.set_mode("train")
        no_better_validation_step = 0

        for i in range(self.start_epoch, self.start_epoch + self.max_epoch):
            start = time.time()
            self.current_epoch = i + 1
            self.train_data_iterator.begin(do_shuffle=True)
            self.wind_train_iterator.begin(do_shuffle=True)
            #logger.info("Epoch: " + str(self.current_epoch) + "/" + str(self.max_epoch))
            train_log.info("Epoch: " + str(self.current_epoch) +
                           "/" + str(self.max_epoch))

            while True:
                if self.verbose:
                    quick_timed_log_eval(logger.debug, "    Gradient Norm:", self.grad_norm_func,
                                         *([self.train_data_iterator.input_batch() + self.wind_train_iterator.input_batch()] +
                                           self.train_data_iterator.output_batch()))
                #print('train data:', np.array(self.train_data_iterator.input_batch()).shape, np.array(self.train_data_iterator.output_batch()).shape, np.array(self.wind_train_iterator.input_batch()).shape, np.array(self.learning_param()).shape)

                '''minibatch_cost = quick_timed_log_eval(logger.debug, "Minibatch Cost:", self.update_func,
                                     *(self.train_data_iterator.input_batch() +
                                       self.train_data_iterator.output_batch() + self.wind_train_iterator.input_batch() +
                                       self.learning_param()))'''
                minibatch_cost = self.update_func(*(self.train_data_iterator.input_batch(
                ) + self.train_data_iterator.output_batch() + self.wind_train_iterator.input_batch() + self.learning_param()))

                # self.update_func()
                self.train_data_iterator.next()
                self.wind_train_iterator.next()
                if self.train_data_iterator.no_batch_left():
                    break

            # quick_timed_log_eval(logger.info, "Training Cost", self.model.get_cost, self.train_data_iterator)
            self.current_validation_error = quick_timed_log_eval(logger.info, "Validation Cost", self.model.get_cost,
                                                                 self.valid_data_iterator, self.wind_valid_iterator)

            logger.info(i, 'validation cost:', self.current_validation_error)
            if len(self.model.error_func_dict) > 0:
                quick_timed_log_eval(logger.info, "Validation Error List", self.model.get_error_dict,
                                     self.valid_data_iterator)
            self.autosave(self.autosave_mode)
            no_better_validation_step += 1
            if self.current_validation_error < self.best_validation_error:
                self.best_validation_error = self.current_validation_error
                no_better_validation_step = 0
            end = time.time()
            if no_better_validation_step >= self.max_epochs_no_best:
                break
            #logger.info("Total Duration For Epoch " + str(self.current_epoch) + ":" + str(end - start))
            train_log.info("Total Duration For Epoch " +
                           str(self.current_epoch) + " : " + str(round((end - start) / 3600, 2)) + "")

    def combine_predict(self, config, save_path):
        self.model.set_mode("predict")
        self.test_data_iterator.begin(do_shuffle=False)
        self.wind_test_iterator.begin(do_shuffle=False)

        # interface symbols: interface.input + interface.output
        predict_func = theano.function(inputs=self.model.interface_layer.input_symbols() + self.model.combine_interface_layer.input_symbols(),
                                       outputs=quick_reshape_patch_back(
                                           self.model.middle_layers[-1].output, config['patch_size']),
                                       on_unused_input='ignore')
        #theano.printing.pydotprint(predict_func, outfile="result/logreg_pydotprint_prediction.png", var_with_name_simple=True)

        result = predict_func(
            *(self.test_data_iterator.input_batch() + self.wind_test_iterator.input_batch()))
        data_max, data_min = self.test_data_iterator.data['max'], self.test_data_iterator.data['min']
        for i, r in enumerate(result):
            image = numpy.reshape(
                r[0], (1, config['size'][0], config['size'][1]))[0]
            image = (image - image.min()) / (image.max() - image.min())

            numpy.savetxt(os.path.join(
                save_path, 'stats' + str(i) + '.out'), image)
            load.plot_map(image, save=True, name=os.path.join(
                save_path, 'predict' + str(i) + '.png'), max=data_max, min=data_min)

            # plot the output batch
            output_image = np.reshape(self.test_data_iterator.output_batch()[
                                      0][i][0], (1, config['size'][0], config['size'][1]))[0]
            load.plot_map(output_image, save=True, name=os.path.join(
                save_path, 'qoutput' + str(i) + '.png'), max=data_max, min=data_min)

        for i in range(config['input_seq_length']):
            # plot the input batch
            input_image = np.reshape(self.test_data_iterator.input_batch()[
                                     0][i][0], (1, config['size'][0], config['size'][1]))[0]
            load.plot_map(input_image, save=True, name=os.path.join(
                save_path, 'input' + str(i) + '.png'), max=data_max, min=data_min)

    def predict(self, config, save_path):
        self.model.set_mode("predict")
        self.test_data_iterator.begin(do_shuffle=False)

        # interface symbols: interface.input + interface.output
        pre = theano.function(inputs=self.model.interface_layer.input_symbols(),
                              outputs=quick_reshape_patch_back(
                                  self.model.middle_layers[-1].output, config['patch_size']),
                              on_unused_input='ignore')
        result = pre(*(self.test_data_iterator.input_batch()))

        # prediction (10, 1, 1, 100, 100), (1, 10, 1, 1, 100, 100)
        #print('prediction result is :', np.array(result).shape, np.array(self.test_data_iterator.input_batch()).shape), np.array(result).max(), np.array(result).min()
        data_max, data_min = self.test_data_iterator.data['max'], self.test_data_iterator.data['min']

        # plot the result and output sequence, using image reshape
        for i, r in enumerate(result):
            image = numpy.reshape(
                r[0], (1, config['size'][0], config['size'][1]))[0]
            image = (image - image.min()) / (image.max() - image.min())
            # image = image*(data_max-data_min)+data_min

            # print image.max(), image.min(), data_max, data_min, config['cmap']

            numpy.savetxt(os.path.join(save_path, 'stats' + str(i) +
                                       '.out'), image * (data_max - data_min) + data_min)
            load.plot_map(image, save=True, name=os.path.join(save_path, 'predict' + str(
                i) + '.png'), max=data_max, min=data_min, vmax=config['vmax'], cmap=config['src'])

            # plot the output batch
            output_image = np.reshape(self.test_data_iterator.output_batch()[
                                      0][i][0], (1, config['size'][0], config['size'][1]))[0]
            load.plot_map(output_image, save=True, name=os.path.join(save_path, 'qoutput' + str(
                i) + '.png'), max=data_max, min=data_min, vmax=config['vmax'], cmap=config['src'])

        # plot the input sequence
        for i in range(config['input_seq_length']):
            # plot the input batch
            input_image = np.reshape(self.test_data_iterator.input_batch()[
                                     0][i][0], (1, config['size'][0], config['size'][1]))[0]
            load.plot_map(input_image, save=True, name=os.path.join(save_path, 'input' + str(
                i) + '.png'), max=data_max, min=data_min, vmax=config['vmax'], cmap=config['src'])

    def train(self, config, save_path):
        process_log.info('***************begin train********')
        train_log.info("***************** begin train *************")
        self.model.set_mode("train")
        no_better_validation_step = 0

        for i in range(self.start_epoch, self.start_epoch + self.max_epoch):
            start = time.time()
            self.current_epoch = i + 1
            self.train_data_iterator0.begin(do_shuffle=True)
            self.train_data_iterator1.begin(do_shuffle=True)
            self.train_data_iterator2.begin(do_shuffle=True)
            self.train_data_iterator3.begin(do_shuffle=True)
            self.train_data_iterator4.begin(do_shuffle=True)
            self.train_data_iterator5.begin(do_shuffle=True)
            #logger.info("Epoch: " + str(self.current_epoch) + "/" + str(self.max_epoch))
            train_log.info("Epoch: " + str(self.current_epoch) +
                           "/" + str(self.max_epoch))

            j = 0
            process_log.info('in ' + str(i + 1) + '/20 while true loop')
            print('in ' + str(i + 1) + '/20 while true loop')
            while True:
                if self.verbose:
                    quick_timed_log_eval(logger.debug, "    Gradient Norm:", self.grad_norm_func,
                                         *(self.train_data_iterator0.input_batch() +
                                           self.train_data_iterator0.output_batch()))
                # print("==============")
                # m = 0
                for i, train_iterator in enumerate([self.train_data_iterator0, self.train_data_iterator1,
                                                    self.train_data_iterator2, self.train_data_iterator3,
                                                    self.train_data_iterator4, self.train_data_iterator5]):
                    print "    -- in " + str(i) + " Train Set..."
                    process_log.info("    -- in " + str(i) + " Train Set, length: " +
                                     str(train_iterator.data['input_raw_data'].shape[0]))
                    while True:
                        minibatch_cost = quick_timed_log_eval(train_log.info, " Minibatch Cost: ", self.update_func,
                                                              *(train_iterator.input_batch() +
                                                                train_iterator.output_batch() +
                                                                self.learning_param()))
                        train_iterator.next()
                        if train_iterator.no_batch_left():
                            break
                    if i == base_config['train_set_num'] - 1:
                        break
                    else:
                        print "    *** Next Train Set ... "
                        process_log.info("    *** Next Train Set ... ")
                j = j + 1
                break
            print "out while True"
            process_log.info("^_^ out while True ")
            #print('valid data:', np.array(self.valid_data_iterator.input_batch()).shape, np.array(self.valid_data_iterator.output_batch()).shape)
            self.current_validation_error = quick_timed_log_eval(train_log.info, "Validation Cost", self.model.get_cost,
                                                                 self.valid_data_iterator)
            self.validation_error_dict['epoch'+str(self.current_epoch)] = self.current_validation_error
            train_log.info(str(self.current_epoch) + ' epoch-validation cost: ' +
                           str(self.current_validation_error))
            process_log.info(str(self.current_epoch) + ' epoch-validation cost: ' +
                             str(self.current_validation_error))
            process_log.info("Current Validation Error Dict: " + self.validation_error_dict)
            if len(self.model.error_func_dict) > 0:
                quick_timed_log_eval(logger.info, "Validation Error List", self.model.get_error_dict,
                                     self.valid_data_iterator)
            self.autosave(self.autosave_mode)
            no_better_validation_step += 1
            if self.current_validation_error < self.best_validation_error:
                self.best_validation_error = self.current_validation_error
                no_better_validation_step = 0
            end = time.time()
            if no_better_validation_step >= self.max_epochs_no_best:
                print " *-_-* Early Stopping !!! "
                process_log.info(" *-_-* Early Stopping !!! ")
                break
            #logger.info("Total Duration For Epoch " + str(self.current_epoch) + ":" + str(end - start))
            train_log.info("Total Duration For Epoch " +
                           str(self.current_epoch) + " : " + str(round((end - start) / 60, 2)) + " mins")

    def autosave(self, mode):
        if "interval" in mode:
            if 0 == (self.current_epoch + 1) % self.save_interval:
                save_path = self.save_path + "/" + self.model.name + "-epoch-" + \
                    str(self.current_epoch) + "-" + \
                    str(self.model.total_param_num()) + ".pkl"
                Model.save(self.model, save_path)
                #logger.info("....Saving to " + os.path.abspath(save_path))
                train_log.info("....Saving to " + os.path.abspath(save_path))
        if "best" in mode:
            if self.current_validation_error < self.best_validation_error:
                save_path = self.save_path + "/" + self.model.name + \
                    str(self.current_epoch) + "-validation-best.pkl"
                Model.save(self.model, save_path)
                #logger.info("....Saving to " + os.path.abspath(save_path))
                train_log.info("....Saving to " + os.path.abspath(save_path))
        if "final" in mode:
            if self.current_epoch == self.max_epoch:
                save_path = self.save_path + "/" + self.model.name + "-epoch-" + \
                    str(self.current_epoch) + "-" + \
                    str(self.model.total_param_num()) + ".pkl"
                Model.save(self.model, save_path)
                #logger.info("....Saving to " + os.path.abspath(save_path))
                train_log.info("....Saving to " + os.path.abspath(save_path))
        if "proceed" in mode:
            if self.current_validation_error < self.best_validation_error:
                save_path = self.save_path + "/" + self.model_name
                Model.save(self.model, save_path)
                #logger.info("....Saving to " + os.path.abspath(save_path))
                train_log.info("....Saving to " + os.path.abspath(save_path))

    def _s(self, s):
        return '%s.%s' % (self.name, s)

    def print_stat(self):
        logger.info("Optimizer Name: " + self.name)
        logger.info("   Common Parameters: ")
        logger.info("      Max Epoch: " + str(self.max_epoch))
        logger.info("      Start Epoch: " + str(self.start_epoch))
        logger.info("      Autosave Mode: " + str(self.autosave_mode))
        logger.info("      Save Interval: " + str(self.save_interval))
        logger.info("      Max Epochs No Best: " +
                    str(self.max_epochs_no_best))
