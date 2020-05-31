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
from coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_file import get_data_file
from neuron_coverage import neuron_coverage
from surprise_coverage import sc
from multi_testing_criteria import multi_testing_criteria
import os
# models=['lenet1', 'lenet4', 'lenet5', 'sub', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
# datasets = 'cifar10'
# datasets='mnist'
# attacks = ['fgsm']
# file = 'cifar10_vgg11_20000_20000_1000'
# epoches = [16,17,18,19,20,21,22]
epoches = range(23)
# file = 'cifar10_vgg11_8000_8000_1000'
file = 'mnist_lenet4_1500_1500_500'
# file = 'cifar10_lenet4_6000_6000_1000'
for epoch in epoches:
    string = file.split('_')
    datasets = string[0]
    model = string[1]

    factor = string[2] + '_' + string[3] + '_' + string[4]
    train_num = int(string[-3])
    test_num = int(string[-1])
    test_num = 1000
    f_raw = open('../draw_picture/data/' + file+'_'+ str(test_num)+ '.txt', 'a')
    # f_raw = open('../draw_picture/data/' + file+ '.txt', 'a')
    print(file+'_'+str(test_num))
    if  os.path.isdir('../multi_testing_criteria/'+datasets+'/'+model):
        shutil.rmtree('../multi_testing_criteria/'+datasets+'/'+model)
    # if  os.path.isdir('../surprise/'+datasets+'/'+ model):
    #     shutil.rmtree('../surprise/'+datasets+'/'+ model)
    # lsa_mean, lsa_std, cov_lsa_1, cov_lsa_2, upper_lsa_test, lower_lsa_test, \
    # dsa_mean, dsa_std, cov_dsa_1, cov_dsa_2, upper_dsa_test, lower_dsa_test = \
    #     sc(datasets=datasets, model_name=model, samples_path='test', layer=-3, num_section=1000, epoch=epoch,
    #        others=factor, test_num=test_num, train_num=train_num)
    neuron_number, result_test = neuron_coverage(datasets=datasets, model_name=model, samples_path='test',
                                                 epoch=epoch, others=factor, test_num=test_num)
    kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets=datasets, model_name=model, samples_path='test',
                                                      std_range=0.0, k_n=1000, k_l=2, epoch=epoch,
                                                      others=factor, test_num=test_num, train_num=train_num)


    f_raw.write('dataset_test: ' + datasets + ' model_factor: ' + model + '_' + factor + ' epoch: ' + str(epoch)
                + ' neuron_number: ' + str(neuron_number) + ' neuron_coverage: ' + str(result_test)
                + ' KMN: ' + str(kmn) + ' NB: ' + str(nb) + ' SNA: ' + str(sna) + ' TKNC: ' + str(tknc) + ' TKNP: ' + str(tknp))
                # + ' lsa_mean: ' + str(lsa_mean) + ' lsa_std: ' + str(lsa_std) + ' dsa_mean: ' + str(dsa_mean) + ' dsa_std: ' + str(dsa_std))
    f_raw.write('\n')
    #
    # lsa_mean, lsa_std, cov_lsa_1, cov_lsa_2, upper_lsa_test, lower_lsa_test, \
    # dsa_mean, dsa_std, cov_dsa_1, cov_dsa_2, upper_dsa_test, lower_dsa_test = \
    #     sc(datasets=datasets, model_name=model, samples_path='test', layer=-3, num_section=1000, epoch=epoch,
    #        datasettype='train', others=factor, train_num=train_num)

    # neuron_number,result_test = neuron_coverage(datasets=datasets, model_name=model, samples_path='test', epoch=epoch,datasettype='train', others = factor, train_num=train_num)
    # kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets=datasets, model_name=model, samples_path='test',
    #                                                   std_range=0.0, k_n=1000, k_l=2, epoch=epoch,datasettype='train', others = factor,train_num=train_num)
    #
    # f_raw.write( 'dataset_train: ' + datasets + ' model_factor: ' + model + '_' + factor + ' epoch: ' + str(epoch)
    #             + ' neuron_number: ' + str(neuron_number) + ' neuron_coverage: ' + str(result_test)
    #             + ' KMN: ' + str(kmn) + ' NB: ' + str(nb) + ' SNA: ' + str(sna) + ' TKNC: ' + str(tknc) + ' TKNP: ' + str(tknp))
    #             # + ' lsa_mean: ' + str(lsa_mean) + ' lsa_std: ' + str(lsa_std) + ' dsa_mean: ' + str(dsa_mean) + ' dsa_std: ' + str(dsa_std))
    # f_raw.write('\n')

# f = open('cifar_coverage_vgg11fgsm.txt', 'w')
# f_raw = open('coverage.txt', 'a')

# for model in ['googlenet16']:
#     '''
#     result_ori_test=neuron_coverage(datasets, model, 'test', epoch=484)
#     f.write(datasets+' neuron coverage using X_test of original model: '+model+' is: '+str(result_ori_test))
#     f.write('\n')
    # '''
    # for attack in attacks:
    #     samples_path = '../adv_result/' + datasets + '/' + attack + '/' + model + '/test_data'
    #     '''
    #     result_ori_attack=neuron_coverage(datasets, model, samples_path, epoch=484)
    #     f.write(datasets+' neuron coverage using combined '+attack+' examples of original model '+model+' is: '+str(result_ori_attack))
    #     f.write('\n')
    #     '''
    #     result_adv_test = neuron_coverage(datasets, model, 'test', de=True, attack=attack, epoch=99)
    #     f.write(
    #         datasets + ' neuron coverage using X_test and ' + attack + ' adversarial training model: ' + model + ' is: ' + str(
    #             result_adv_test))
    #     f.write('\n')
    #
    #     result_adv_attack = neuron_coverage(datasets, model, samples_path, de=True, attack=attack, epoch=99)
    #     f.write(
    #         datasets + ' neuron coverage using combined ' + attack + ' examples and ' + attack + ' adversarial training model: ' + model + ' is: ' + str(
    #             result_adv_attack))
    #     f.write('\n')
    #
    #     '''
    #     result_ori_justadv=neuron_coverage(datasets, model, samples_path, just_adv=True, epoch=49)
    #     f.write(datasets+' neuron coverage using just '+attack+' examples of original model '+model+' is: '+str(result_ori_justadv))
    #     f.write('\n')
    #     result_adv_justadv=neuron_coverage(datasets, model, samples_path, de=True, attack=attack, just_adv=True, epoch=49)
    #     f.write(datasets+' neuron coverage using just '+attack+' examples and '+attack+' adversarial training model: '+model+' is: '+str(result_adv_justadv))
    #     f.write('\n')
    #     '''
    # '''
    # kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, 'test', epoch=484)
    # f.write('datasets: '+datasets+' model: orignal '+model+' test_set: X_test:')
    # f.write('\n')
    # f.write('KMN: '+str(kmn))
    # f.write('\n')
    # f.write('NB: '+str(nb))
    # f.write('\n')
    # f.write('SNA: '+str(sna))
    # f.write('\n')
    # f.write('TKNC: '+str(tknc))
    # f.write('\n')
    # f.write('TKNP: '+str(tknp))
    # f.write('\n')
    # '''
    # for attack in attacks:
    #     samples_path = '../adv_result/' + datasets + '/' + attack + '/' + model + '/test_data'
    #     '''
    #     kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, epoch=484)
    #     f.write('datasets: '+datasets+' model: orignal '+model+' test_set: combined '+attack+' adv example:')
    #     f.write('\n')
    #     f.write('KMN: '+str(kmn))
    #     f.write('\n')
    #     f.write('NB: '+str(nb))
    #     f.write('\n')
    #     f.write('SNA: '+str(sna))
    #     f.write('\n')
    #     f.write('TKNC: '+str(tknc))
    #     f.write('\n')
    #     f.write('TKNP: '+str(tknp))
    #     f.write('\n')
    #     '''
    #     kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets, model, 'test', de=True, attack=attack, epoch=99)
    #     f.write('datasets: ' + datasets + ' model: ' + attack + ' adv_training ' + model + ' test_set: X_test:')
    #     f.write('\n')
    #     f.write('KMN: ' + str(kmn))
    #     f.write('\n')
    #     f.write('NB: ' + str(nb))
    #     f.write('\n')
    #     f.write('SNA: ' + str(sna))
    #     f.write('\n')
    #     f.write('TKNC: ' + str(tknc))
    #     f.write('\n')
    #     f.write('TKNP: ' + str(tknp))
    #     f.write('\n')
    #
    #     kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets, model, samples_path, de=True, attack=attack,
    #                                                       epoch=99)
    #
    #     f.write(
    #         'datasets: ' + datasets + ' model: ' + attack + ' adv_training ' + model + ' test_set: combined ' + attack + ' adv example:')
    #     f.write('\n')
    #     f.write('KMN: ' + str(kmn))
    #     f.write('\n')
    #     f.write('NB: ' + str(nb))
    #     f.write('\n')
    #     f.write('SNA: ' + str(sna))
    #     f.write('\n')
    #     f.write('TKNC: ' + str(tknc))
    #     f.write('\n')
    #     f.write('TKNP: ' + str(tknp))
    #     f.write('\n')
    #
    #     '''
    #     kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, just_adv=True, epoch=49)
    #     f.write('datasets: '+datasets+' model: orignal '+model+' test_set: just '+attack+' adv example:')
    #     f.write('\n')
    #     f.write('KMN: '+str(kmn))
    #     f.write('\n')
    #     f.write('NB: '+str(nb))
    #     f.write('\n')
    #     f.write('SNA: '+str(sna))
    #     f.write('\n')
    #     f.write('TKNC: '+str(tknc))
    #     f.write('\n')
    #     f.write('TKNP: '+str(tknp))
    #     f.write('\n')
    #
    #     kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, de=True, attack=attack, just_adv=True, epoch=49)
    #
    #     f.write('datasets: '+datasets+' model: '+attack+' adv_training '+model+' test_set: just '+attack+' adv example:')
    #     f.write('\n')
    #     f.write('KMN: '+str(kmn))
    #     f.write('\n')
    #     f.write('NB: '+str(nb))
    #     f.write('\n')
    #     f.write('SNA: '+str(sna))
    #     f.write('\n')
    #     f.write('TKNC: '+str(tknc))
    #     f.write('\n')
    #     f.write('TKNP: '+str(tknp))
    #     f.write('\n')
    #     '''

#记录测试数据集在所有模型的神经元覆盖率 模型文件目录：/20200313/DRTest/models/cifar10_googlenet16/4

# for file in os.listdir('../models'):
#
#     if not os.path.isdir(file):
#
#         if file != 'cifar10_lenet4_20000_20000_1000':
#             continue
#
#         f_raw = open('../draw_picture/data' + file + '.txt', 'a')
#         string = file.split('_')
#         datasets = string[0]
#         model = string[1]
#
#         factor = string[2] + '_' + string[3]+'_'+string[4]
#         train_num = int(string[-3])
#         test_num = int(string[-1])
#
#         for epoch in os.listdir('../models_test/'+file):
#             if epoch== 'logs':
#                 continue
#             epoch = int(epoch)
#             if not os.path.isdir('../multi_testing_criteria'):
#                 shutil.rmtree('../multi_testing_criteria')
#             if not os.path.isdir('../surprise'):
#                 shutil.rmtree('../surprise')
#             neuron_number, result_test = neuron_coverage(datasets=datasets, model_name=model, samples_path='test', epoch=epoch, others = factor,test_num=test_num)
#             kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets= datasets, model_name= model, samples_path= 'test',
#                                    std_range=0.0, k_n=1000, k_l= 2, epoch = epoch, others = factor, test_num=test_num,train_num=train_num)
#             lsa_mean, lsa_std, cov_lsa_1, cov_lsa_2, upper_lsa_test, lower_lsa_test, \
#             dsa_mean, dsa_std, cov_dsa_1, cov_dsa_2, upper_dsa_test, lower_dsa_test = \
#                 sc(datasets=datasets, model_name=model, samples_path='test', layer=-3, num_section=1000, epoch=epoch,
#                    others = factor,test_num=test_num,train_num=train_num)
#
#
#
#             f_raw.write('dataset_test: ' + datasets + ' model_factor: ' + model + '_' + factor + ' epoch: ' + str(epoch)
#                        + ' neuron_number: ' + str(neuron_number) + ' neuron_coverage: ' + str(result_test)
#                        + ' KMN: ' + str(kmn) + ' NB: ' + str(nb) + ' SNA: ' + str(sna) + ' TKNC: ' + str(tknc) + ' TKNP: ' + str(tknp)
#                        + ' lsa_mean: ' + str(lsa_mean) + ' lsa_std: ' + str(lsa_std) + ' dsa_mean: ' + str(dsa_mean) + ' dsa_std: ' + str(dsa_std))
#             f_raw.write('\n')
#             # shutil.rmtree('../multi_testing_criteria/dataset/model/ori/test')
#             # shutil.rmtree('../surprise/test')
#             # shutil.rmtree('../surprise/train')
#             #
#             # neuron_number,result_test = neuron_coverage(datasets=datasets, model_name=model, samples_path='test', epoch=epoch,datasettype='train', others = factor, train_num=train_num)
#             # kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets=datasets, model_name=model, samples_path='test',
#             #                                                   std_range=0.0, k_n=1000, k_l=2, epoch=epoch,datasettype='train', others = factor,train_num=train_num)
#             # lsa_mean, lsa_std, cov_lsa_1, cov_lsa_2, upper_lsa_test, lower_lsa_test, \
#             # dsa_mean, dsa_std, cov_dsa_1, cov_dsa_2, upper_dsa_test, lower_dsa_test = \
#             #     sc(datasets=datasets, model_name=model, samples_path='test', layer=-3, num_section=1000, epoch=epoch,
#             #        datasettype='train', others = factor,train_num=train_num)
#             #
#             # f_raw.write( 'dataset_train: ' + datasets + ' model_factor: ' + model + '_' + factor + ' epoch: ' + str(epoch)
#             #             + ' neuron_number: ' + str(neuron_number) + ' neuron_coverage: ' + str(result_test)
#             #             + ' KMN: ' + str(kmn) + ' NB: ' + str(nb) + ' SNA: ' + str(sna) + ' TKNC: ' + str(tknc) + ' TKNP: ' + str(tknp)
#             #             + ' lsa_mean: ' + str(lsa_mean) + ' lsa_std: ' + str(lsa_std) + ' dsa_mean: ' + str(dsa_mean) + ' dsa_std: ' + str(dsa_std))
#             # f_raw.write('\n')
#     f_raw.close()



