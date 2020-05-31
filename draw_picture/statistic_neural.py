import sys
import os
import re
import xlwt
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

sys.path.append("../")

def read_file(file_name):

    path ='../draw_picture/ndata/'+file_name
    f = open(path, 'r')
    #dic{epoch:layer{neuron:}}
    dictionary = {}
    for line_raw in f:
        dic_layer = {}
        string1 = re.search(r'epoch: (.+) neuron_number', line_raw).group()
        epoch = int(string1.replace('epoch: ','').replace(' neuron_number',''))
        string2 = re.search(r"[[][(](.*)[)][]]", line_raw).group().replace('[','').replace(']','').replace('(','').replace(')','').replace(' ','').split(',')
        for i in range(int(len(string2)/3)):
            i = i * 3
            if string2[i] not in dic_layer.keys():
                dic_layer[string2[i]] = {}
            dic_layer[string2[i]][string2[i+1]] = int(string2[i+2])
        dictionary[epoch] = dic_layer
    return dictionary

def write_xls(file_name):
    #统计层写入xls中
    dictionary = read_file(file_name)
    test_num = int(file_name.split('_')[-1].replace('.txt',''))
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    #{epoch:{[layer_full],[layer_half],[layer_absense]}
    epoch_dict = {}
    for i in range(len(dictionary)):
        epoch = dictionary[i]
        epoch_dict[i]=[]
        layer_full = []
        layer_half = []
        layer_absense = []
        for j in epoch.keys():
            layer = epoch[j]
            count = 0
            for k in layer.keys():
                count += layer[k]
                if layer[k] != test_num and layer[k] > 0 :
                    layer_half.append(j)
                    break

            if count == 0:
                layer_absense.append(j)
            if count == test_num * len(layer):
                layer_full.append(j)
        epoch_dict[i].append(layer_full)
        epoch_dict[i].append(layer_half)
        epoch_dict[i].append(layer_absense)

    for i in range(len(epoch_dict)):
        for j in range(3):
            booksheet.write(i, j,  str(epoch_dict[i][j]).replace('[','').replace(']',''))
    workbook.save('../draw_picture/st_ndata/xls/'+file_name.replace('.txt','')+'.xls')

def write_txt(file_name):
    #统计每一层中重复了testnum的神经元写入txt中
    dictionary = read_file(file_name)
    test_num = int(file_name.split('_')[-1].replace('.txt',''))
    f = open('../draw_picture/st_ndata/txt/'+file_name,'a')
    for i in range(len(dictionary)):
        epoch = dictionary[i]
        f.write('epoch: ' + str(i))
        for j in epoch.keys():
            layer = epoch[j]
            neuron = []
            for k in layer.keys():
                if layer[k] == test_num:
                    neuron.append(k)
            f.write( ' layer: ' + j + ' neuron: '+ str(neuron))
        f.write('\n')
    f.close()

def write_layer_rate(file_name):
    #统计每个epoch中每一层的神经元重复比率
    dictionary = read_file(file_name)
    test_num = int(file_name.split('_')[-1].replace('.txt',''))
    f = open('../draw_picture/st_ndata/layer_rate/'+file_name,'a')
    for i in range(len(dictionary)):
        epoch = dictionary[i]
        f.write('epoch: ' + str(i))
        for j in epoch.keys():
            layer = epoch[j]
            neuron = []
            count = 0
            for k in layer.keys():
                count += layer[k]
            f.write( ' layer: ' + j + ' rate: '+ str(count/(len(layer.keys())*test_num)))
        f.write('\n')
    f.close()

def write_dul_rate(file_name):
    #统计每个epoch中每一层的神经元重复比率
    dictionary = read_file(file_name)
    test_num = int(file_name.split('_')[-1].replace('.txt',''))
    f = open('../draw_picture/st_ndata/dul/'+file_name,'a')
    for i in range(len(dictionary)):
        epoch = dictionary[i]
        f.write('epoch: ' + str(i))
        count = 0
        count_neuron = 0
        for j in epoch.keys():
            layer = epoch[j]
            for k in layer.keys():
                count += layer[k]
            count_neuron += len(layer.keys())*test_num

        f.write( ' rate: '+ str(count/count_neuron))
        f.write('\n')
    f.close()

def draw_layer_rate(file_name):
    path = '../draw_picture/st_ndata/layer_rate/' + file_name
    f = open(path, 'r')
    dictionary = {}
    count = 0
    for line_raw in f:
        count += 1
        layer_list  = re.findall(r'layer: (.+?) rate:', line_raw)
        rate_list = re.findall(r'rate: (\d+\.?\d*)', line_raw)
        for i in range(len(layer_list)):
            layer = layer_list[i]
            if layer not in dictionary.keys():
                dictionary[layer] = []
            if len(rate_list[i]) > 5:
                rate = rate_list[i][:5]
            else:
                rate = rate_list[i]
            dictionary[layer].append(float((rate)))
    plt.clf()
    plt.figure(figsize=(20,5))
    my_x_ticks = np.arange(0, 1, 0.05)
    plt.xticks(my_x_ticks)
    x = [i for i in range(count)]
    for layer in dictionary.keys():
        plt.plot(dictionary[layer],x, label = layer)

    plt.legend(loc='lower right',prop={'size': 10})
    plt.savefig('../draw_picture/st_ndata/layer_rate_picture/' + file_name.replace('.txt','') + '.jpg')

def main():
    file_name = 'cifar10_resnet18_50000_50000_1000_1000.txt'
    # file_name = 'mnist_resnet18_100_100_500_1000.txt'

    # write_xls
    # write_dul_rate(file_name)
    # write_layer_rate(file_name)
    draw_layer_rate(file_name)


if __name__ == '__main__':
    main()


