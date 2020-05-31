from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load
from statistic.utils import init_coverage_tables, neuron_covered, update_coverage
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_file import get_data_file
from statistic_neuron_coverage import neuron_coverage
from statistic_multi_testing_criteria import multi_testing_criteria
import os
# models=['lenet1', 'lenet4', 'lenet5', 'sub', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
# datasets = 'cifar10'
# datasets='mnist'
# attacks = ['fgsm']
# file = 'cifar10_vgg11_20000_20000_1000'
# epoches = [16,17,18,19,20,21,22]
epoches = range(23)
file = 'cifar10_resnet18_20000_20000_800'
# file = 'mnist_lenet4_1500_1500_500'
# file = 'cifar10_lenet4_6000_6000_1000'
for epoch in epoches:
    string = file.split('_')
    datasets = string[0]
    model = string[1]

    factor = string[2] + '_' + string[3] + '_' + string[4]
    train_num = int(string[-3])
    test_num = int(string[-1])
    test_num = 1000
    f_raw = open('../draw_picture/ndata/' + file+'_'+ str(test_num)+ '.txt', 'a')
    # f_raw = open('../draw_picture/data/' + file+ '.txt', 'a')
    print(file+'_'+str(test_num))
    if  os.path.isdir('../multi_testing_criteria/'+datasets+'/'+model):
        shutil.rmtree('../multi_testing_criteria/'+datasets+'/'+model)

    neuron_number, result_test, tenpercent = neuron_coverage(datasets=datasets, model_name=model, samples_path='test',
                                                 epoch=epoch, others=factor, test_num=test_num)
    # kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets=datasets, model_name=model, samples_path='test',
    #                                                   std_range=0.0, k_n=1000, k_l=2, epoch=epoch,
    #                                                   others=factor, test_num=test_num, train_num=train_num)

    f_raw.write('dataset_test: ' + datasets + ' model_factor: ' + model + '_' + factor + ' epoch: ' + str(epoch)
                + ' neuron_number: ' + str(neuron_number) + ' neuron_coverage: ' + str(tenpercent)+ ' result_show: '+str(result_test))
                # + ' KMN: ' + str(kmn) + ' NB: ' + str(nb) + ' SNA: ' + str(sna) + ' TKNC: ' + str(tknc) + ' TKNP: ' + str(tknp))
    f_raw.write('\n')