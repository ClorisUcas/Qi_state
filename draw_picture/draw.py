import sys
import os
from tensorboard.backend.event_processing import event_accumulator

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

sys.path.append("../")

def read_acc(file_acc,file_acc_new='default'):
    # 加载日志数据
    dictionary = {}
    string = file_acc.split('_')
    datasets = string[0]
    factor = file_acc.replace(datasets+'_','')

    if file_acc_new == 'default':
        # paths = ['../models/'+file_acc+'/logs/train','../models/'+file_acc+'/logs/test']
        paths = ['../models/'+file_acc+'/logs/train','../models/'+file_acc+'/logs/val']
    else:
        # paths = ['../models/' + file_acc + '/logs/train', '../models_test/' + file_acc_new + '/logs/test']
        paths = ['../models/' + file_acc + '/logs/train', '../models_test/' + file_acc_new + '/logs/val']

    for path in paths:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()

        data = path.split('/')[-1]
        # 是train test文件夹 (还有一个train val)
        # if data == 'train':
        #     data = 'val'
        #     data1 = 'train'
        val_acc = ea.scalars.Items(data+'/acc')
        val_loss = ea.scalars.Items(data + '/loss')
        acc =  [i.value for i in val_acc]
        loss =  [i.value for i in val_loss]

        if datasets not in dictionary.keys():
            dictionary[datasets] = {}
        if factor not in dictionary[datasets].keys():
            dictionary[datasets][factor] = {}
        #第一组是train val文件夹，第二组是train test文件夹
        if data == 'train':
        # if data == 'val':
            if 'dataset_train' not in dictionary[datasets][factor].keys():
                dictionary[datasets][factor]['dataset_train'] = {}
            dictionary[datasets][factor]['dataset_train'] = [acc,loss]
        if data == 'val':
        # if data == 'test':
            if 'dataset_test' not in dictionary[datasets][factor].keys():
                dictionary[datasets][factor]['dataset_test'] = {}
            dictionary[datasets][factor]['dataset_test'] = [acc,loss]

    return dictionary

def read_cov(filename,file_new = 'default'):
    #file_new是为了读取新的test集的cov
    # dict{'mnist':{lenet_datasize_:{dataset_type:{neuron:[]}}}
    # }
    dict = {}
    dictionary = {}
    for file in os.listdir('../draw_picture/data/'):
        if file == filename or file == file_new:
            f = open('../draw_picture/data/'+file, 'r')
            for line_raw in f:
                line_raw = line_raw.replace(':','')
                line = line_raw.split(' ')
                # datasets_train: mnist model_factor: lenet1_datasize_60000 epoch: 32 neuron_number: 68 neuron coverage: 1.0
                # KMN: 0.767235294117647 NB: 0.0 SNA: 0.0 TKNC: 0.8970588235294118 TKNP: 5533 lsa_mean: 0.09723266639417098
                # lsa_std: 0.11946089727559596 dsa_mean: 0.313298284627805 dsa_std: 0.13436943451500136
                cov = {}
                if file_new != 'default' :
                    if file == filename:
                        if line[0] == 'dataset_test': continue

                for i in [6, 8, 10, 12, 14, 16, 18]:
                    cov[line[i]] = float(line[i + 1])

                if line[1] not in dict.keys():
                    dict[line[1]] = {}
                if line[3] not in dict[line[1]].keys():
                    dict[line[1]][line[3]] = {}
                if line[0] not in dict[line[1]][line[3]].keys():
                    dict[line[1]][line[3]][line[0]] = {}
                if dictionary.keys() == []:
                    dictionary = dict
                dict[line[1]][line[3]][line[0]][int(line[5])] = cov
            f.close()
            # 整合dict
    dictionary = dict
    for dataset in dict.keys():
        for model in dict[dataset].keys():
            for dataset_type  in dict[dataset][model].keys():
                epochs = dict[dataset][model][dataset_type]
                dict_cov ={}
                dict_cov['neuron_coverage'] = []
                dict_cov['KMN'] = []
                dict_cov['NB'] = []
                dict_cov['SNA'] = []
                dict_cov['TKNC'] = []
                # dict_cov['lsa_mean'] = []
                # dict_cov['dsa_mean'] = []

                epoch_number = len(epochs)
                for i in range(epoch_number):

                    dict_cov['neuron_coverage'].append(epochs[i]['neuron_coverage'])
                    dict_cov['KMN'].append(epochs[i]['KMN'])
                    dict_cov['NB'].append(epochs[i]['NB'])
                    dict_cov['TKNC'].append(epochs[i]['TKNC'])
                    dict_cov['SNA'].append(epochs[i]['SNA'])
                    # dict_cov['lsa_mean'].append(epochs[i]['lsa_mean'])
                    # dict_cov['dsa_mean'].append(epochs[i]['dsa_mean'])
                dictionary[dataset][model][dataset_type] = dict_cov
    return dictionary

def draw_pic(file_acc,file_cov,file_acc_new='default', file_cov_new='default'):
    dic_acc = read_acc(file_acc,file_acc_new)
    dic_cov = read_cov(file_cov,file_cov_new)

    for dataset in dic_cov.keys():
        for model  in dic_cov[dataset].keys():
            plt.clf()
            plt.figure(23)
            for dataset_type in dic_cov[dataset][model].keys():
                if dataset_type == 'dataset_train':
                    cov_label = 'train'
                else:
                    cov_label = 'test'
                temper = dic_cov[dataset][model][dataset_type]
                temper_acc =  dic_acc[dataset][model][dataset_type][0]
                temper_loss = dic_acc[dataset][model][dataset_type][1]

                neuron_coverage =  temper['neuron_coverage']
                KMN = temper['KMN']
                NB = temper['NB']
                TKNC = temper['TKNC']
                SNA = temper['SNA']
                # lsa_mean = temper['lsa_mean']
                # dsa_mean = temper['dsa_mean']
                number = len(neuron_coverage)
                x = [i for i in range(number)]
                temper_acc = temper_acc[:number]
                temper_loss = temper_loss[:number]
                plt.subplot(231)
                plt.plot(x, neuron_coverage,label = 'nc/'+cov_label)
                # plt.plot(x, temper_acc,label = 'acc/'+cov_label)
                # plt.plot(x, temper_loss, label='loss/' + cov_label)
                plt.legend(loc='lower right',prop={'size':8})

                plt.subplot(232)
                plt.plot(x, KMN,label = 'KMN/'+cov_label)
                # plt.plot(x, temper_acc, label='acc/' + cov_label)
                # plt.plot(x, temper_loss, label='loss/' + cov_label)
                plt.legend(loc='lower right',prop={'size': 8})

                plt.subplot(233)
                plt.plot(x, NB,label = 'NB/'+cov_label)
                # plt.plot(x, temper_acc, label='acc/' + cov_label)
                # plt.plot(x, temper_loss, label='loss/' + cov_label)
                plt.legend(loc='lower right',prop={'size':8})

                plt.subplot(234)
                plt.plot(x, TKNC,label = 'TKNC/'+cov_label)
                # plt.plot(x, temper_acc, label='acc/' + cov_label)
                # plt.plot(x, temper_loss, label='loss/' + cov_label)
                plt.legend(loc='lower right',prop={'size':8})

                plt.subplot(235)
                plt.plot(x, SNA ,label = 'SNA/'+cov_label)
                # plt.plot(x, temper_acc, label='acc/' + cov_label)
                # plt.plot(x, temper_loss, label='loss/' + cov_label)
                plt.legend(loc='lower right',prop={'size':8})

                # plt.subplot(236)
                # plt.plot(x, dsa_mean,label = 'dsa_mean/'+cov_label)
                # plt.plot(x, temper_acc, label='acc/' + cov_label)
                # plt.legend(prop={'size':8})
            # 调整每隔子图之间的距离
            plt.tight_layout()
            if file_acc_new == 'default':
                plt.savefig('./picture/'+ file_acc + '.jpg')
            else:
                plt.savefig('./picture/' + file_acc_new + '.jpg')

def write_acc(file_acc,file_acc_new='default'):
    dic_acc = read_acc(file_acc,file_acc_new)

    for dataset in dic_acc.keys():
        for model  in dic_acc[dataset].keys():
            for dataset_type in dic_acc[dataset][model].keys():
                temper_acc =  dic_acc[dataset][model][dataset_type][0]
                temper_loss = dic_acc[dataset][model][dataset_type][1]

                if file_acc_new == 'default':
                    f = open('../draw_picture/loss_acc/' + file_acc+'.txt', 'a')
                else:
                    f = open('../draw_picture/loss_acc/' + file_acc_new+'.txt', 'a')
                f.write(dataset_type + ': ' + ' acc: ' + str(temper_acc) + ' loss: ' + str(temper_loss) + '\n')


#对于训练中断的模型，不再重新训练，可以先画图看结果
def re_draw(filename):
    for file in os.listdir('../models/'):
        if file != filename:
            continue
        string = file.split('_')
        datasets = string[0]
        factor = file.replace(datasets + '_', '')
        fig = plt.figure(figsize=(10, 4))
        for logs in os.listdir('../models/' + file):
            if logs == 'logs':
                for data in os.listdir('../models/' + file + '/logs'):
                    ea = event_accumulator.EventAccumulator(r'../models/' + file + '/logs/' + data)
                    ea.Reload()
                    # if data == 'train':
                    #     data='val'

                    val_loss = ea.scalars.Items(data + '/loss')
                    val_acc = ea.scalars.Items(data + '/acc')

                    ax1 = fig.add_subplot(122)
                    ax1.plot([i.step for i in val_acc], [i.value for i in val_acc], label=data + '/acc')
                    ax1.set_xlim(0)

                    plt.legend(loc='lower right')
                    ax2 = fig.add_subplot(121)
                    ax2.plot([i.step for i in val_loss], [i.value for i in val_loss], label=data + '/loss')
                    ax2.set_xlim(0)
                plt.savefig('../models/'+filename+'/logs/result.jpg')


def main():
    #为了绘制覆盖率的图像和准确率和损失的图像
    # file_acc,file_cov都是为了训练集的准确率和覆盖率，两个new可以测不同测试数据的值
    file_acc = 'cifar10_vgg11m_8000_8000_500'
    file_acc_new = 'default'
    # file_acc_new = 'mnist_lenet4_1500_1500_1000'
    # file_cov  = 'mnist_lenet4_1500_1500_500.txt'
    # file_cov_new = 'default'
    # file_cov_new = 'mnist_lenet4_1500_1500_500_1000.txt'
    # draw_pic(file_acc,file_cov,file_acc_new,file_cov_new)
    write_acc(file_acc,file_acc_new)


    # filename = 'cifar10_resnet18_50000_50000_1000'
    # re_draw(filename)


if __name__ == '__main__':
    main()


