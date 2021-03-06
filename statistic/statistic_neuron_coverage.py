#Pei, Kexin, et al. "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." (2017).#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load
from statistic.utils import init_coverage_tables, neuron_covered, update_coverage
from nmutant_data.data import get_data
from nmutant_util.utils_file import get_data_file

FLAGS = flags.FLAGS

def neuron_coverage(datasets, model_name, samples_path, others='',train_num=0 , test_num=0 ,de=False, attack='fgsm', just_adv=False, epoch=49, datasettype = 'test'):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    if datasettype == 'train':
        samples = X_train[:train_num]
    else:
        samples = X_test[:test_num]

    if samples_path not in ['test']:
        if not just_adv:
            [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
            #samples_adv = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list])

            samples_adv=np.asarray(image_list)
          
            samples = np.concatenate((samples, samples_adv))
            print("Combine data")
        else:
            [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
            #samples_adv = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list])

            samples=np.asarray(image_list)
          
            #samples = np.concatenate((samples, samples_adv))
            print("Just adv")

    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                     epoch=epoch,others=others)
    model_layer_dict = init_coverage_tables(model)
    sess.close()
    del sess, preds, x, y, model, feed_dict
    gc.collect()
    #ceil取整数
    n_batches = int(np.ceil(samples.shape[0] / 256))

    for i in range(n_batches):
        print(i)
        start = i * 256
        end = np.minimum(len(samples), (i + 1) * 256)
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                         epoch=epoch,others=others)
        model_layer_dict = update_coverage(sess, x, samples[start:end], model, model_layer_dict, feed_dict, threshold=0)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()

    tenpercent,result = neuron_covered(model_layer_dict)

    return len(model_layer_dict),result,tenpercent

def main(argv=None):
    neuron_coverage(datasets = FLAGS.datasets,
                    model_name=FLAGS.model,
                    samples_path=FLAGS.samples,
                    epoch=FLAGS.epoch)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')#'../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_string('epoch',4)
    tf.app.run()
