from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
sys.path.append("../")
from nmutant_model.model_operation import model_load
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import stats as sts
# from scipy.optimize import curve_fit
from scipy.misc import derivative

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

FLAGS = flags.FLAGS

def get_filters(sess,model):
    # 获取某个epoch模型中的filter tensor值
    count = 0
    filters = []
    for key in model.layer_names:
        if 'Conv' in key:
            filters.append(sess.run(model.layers[count].kernels.value()))
        count += 1
    return filters

def get_layout(sess, x, input_data, model,  feed_dict):

    layer_outputs = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'Flatten' not in key and 'Input' not in key:
            tensor = dict[key]
            feed = {x: input_data}
            if feed_dict is not None:
                feed.update(feed_dict)
            layer_outputs.append(sess.run(tensor, feed_dict=feed))

    return layer_outputs

def write_ee(file,epochs):
    # 计算不同模型欧式距离/每一个模型的最大filter差距
    string = file.split('_')
    datasets = string[0]
    model_name = string[1]

    factor = string[2] + '_' + string[3] + '_' + string[4]
    filters_dict = {}

    f = open('../filter1/filter_euclidean/' + datasets + '_' + model_name + '_' + factor + '_ee' + '.txt','w')
    epoch_df = {}
    for epoch in range(epochs):
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=False,
                                                         attack='fgsm',
                                                         epoch=epoch, others=factor)
        filters_dict[epoch] = get_filters(sess, model)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()
        x, y = data_process(filters_dict[epoch][0], filters_dict[epoch][-1])
        epoch_df[epoch] = np.linalg.norm(x - y)

    layer_epoch = {}
    for key in filters_dict.keys():
        for i in range(len(filters_dict[key])):
            if i not in layer_epoch.keys():
                layer_epoch[i] = []
            layer_epoch[i].append(filters_dict[key][i])

    for key in layer_epoch.keys():
        filters = layer_epoch[key]
        f.write(str(key))
        for i in range(len(filters)):
            f.write(' ' + str(np.linalg.norm(filters[0] - filters[i])/abs(epoch_df[i]-epoch_df[0])))
        f.write('\n')

def write_ee1(file,epochs):
    # 计算每个模型/每一个模型的最后一层和第一层filter差距的欧式距离
    string = file.split('_')
    datasets = string[0]
    model_name = string[1]

    factor = string[2] + '_' + string[3] + '_' + string[4]
    filters_dict = {}

    f = open('../filter1/filter_euclidean/' + datasets + '_' + model_name + '_' + factor + '_ee1' + '.txt','w')
    epoch_df = {}
    for epoch in range(epochs):
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=False,
                                                         attack='fgsm',
                                                         epoch=epoch, others=factor)
        filters_dict[epoch] = get_filters(sess, model)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()
        x, y = data_process(filters_dict[epoch][0], filters_dict[epoch][-1])
        epoch_df[epoch] = np.linalg.norm(x - y)

    layer_epoch = {}
    for key in filters_dict.keys():
        for i in range(len(filters_dict[key])):
            if i not in layer_epoch.keys():
                layer_epoch[i] = []
            layer_epoch[i].append(filters_dict[key][i])

    for key in layer_epoch.keys():
        filters = layer_epoch[key]
        f.write(str(key))
        for i in range(len(filters)):
            f.write(' ' + str(np.linalg.norm((filters[0]-abs(epoch_df[0]))/abs(epoch_df[0]) - (filters[i]-abs(epoch_df[i]))/abs(epoch_df[i]))))
        f.write('\n')

def compare_epochs(filter_dict,datasets, model_name,others):
    # 比较不同epoch模型相同层提取的特征的欧式距离
    f = open('../filter1/filter_euclidean/' + datasets + '_' + model_name + '_' + others + '.txt', 'w')
    layer_epoch = {}
    for key in filter_dict.keys():
        for i in range(len(filter_dict[key])):
            if i not in layer_epoch.keys():
                layer_epoch[i] = []
            layer_epoch[i].append(filter_dict[key][i])

    for key in layer_epoch.keys():
        filters = layer_epoch[key]
        f.write(str(key))
        for i in range(len(filters)-1):
            f.write(' '+ str(np.linalg.norm(filters[i]-filters[i+1])))
        f.write('\n')

def data_process(x,y):
    # 计算欧式距离补0
    if x.shape != y.shape:
        if y.shape[0]>x.shape[0]:
            x = np.pad(x, ((0, y.shape[0] - x.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')
        else:
            y = np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')

        if y.shape[1]>x.shape[1]:
            x = np.pad(x, ((0, 0), (0, y.shape[1] - x.shape[1]), (0, 0), (0, 0)), 'constant')
        else:
            y = np.pad(y, ((0, 0), (0, x.shape[1] - y.shape[1]), (0, 0), (0, 0)), 'constant')

        if y.shape[2]>x.shape[2]:
            x = np.pad(x, ((0, 0), (0, 0), (0, y.shape[2] - x.shape[2]), (0, 0)), 'constant')
        else:
            y = np.pad(y, ((0, 0), (0, 0), (0, x.shape[2] - y.shape[2]), (0, 0)), 'constant')

        if y.shape[3]>x.shape[3]:
            x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, y.shape[3] - x.shape[3])), 'constant')
        else:
            y = np.pad(y, ((0, 0), (0, 0), (0, 0), (0, x.shape[3] - y.shape[3])), 'constant')

    return x,y

def write_txt(file,epochs):
    string = file.split('_')
    datasets = string[0]
    model_name = string[1]

    factor = string[2] + '_' + string[3] + '_' + string[4]
    filters_dict = {}

    f = open('../filter1/filter_euclidean/' + datasets + '_' + model_name + '_' + factor + '_' + str(epochs) + '.txt',
             'w')
    f_layers = open('../filter1/filter_euclidean/' + datasets + '_' + model_name + '_' + factor + '_layers' + '.txt',
                    'w')
    for epoch in range(epochs):
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=False,
                                                         attack='fgsm',
                                                         epoch=epoch, others=factor)
        filters_dict[epoch] = get_filters(sess, model)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()
        x, y = data_process(filters_dict[epoch][0], filters_dict[epoch][-1])
        f.write(str(epoch) + ' ' + str(np.linalg.norm(x - y)))
        f.write('\n')

        f_layers.write(str(epoch) )
        for num in range(len(filters_dict[epoch])):
            x, y = data_process(filters_dict[epoch][0], filters_dict[epoch][num])
            f_layers.write(' ' +str(np.linalg.norm(x - y)))
        f_layers.write('\n')

    compare_epochs(filters_dict,datasets=datasets, model_name=model_name,others=factor)

def draw_layers(file):
    # 绘制layers的图
    # 18 16.918877 28.349625 38.03104 46.132736 49.95466 45.669487 42.9263 层与层之间的欧式距离
    plt.figure()
    f = open('../filter1/filter_euclidean/' + file+'_layers.txt', 'r')
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        x = [str(i) for i in range(len(line)-1)]
        y = []
        for i in range(len(line)):
            if i == 0:
                continue
            y.append(float(line[i]))
        plt.plot(x, y, label=line[0])
        plt.legend(loc='best', bbox_to_anchor=(1,1),prop={'size': 8})
        plt.savefig('../filter1/filter_euclidean_pic/' + file + '_layers.jpg')
    f.close()

def draw_point(file):
    # file_name2 = 'mnist_vgg11_6000_6000_500_'+str(epochs)

    # 绘制每一个模型第一个filter和最后一个filter的欧式距离
    plt.figure()
    f = open('../filter1/filter_euclidean/' + file+'.txt', 'r')
    x,y = [],[]
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        x.append(line[0])
        y.append(float(line[1]))
    plt.plot(x, y)
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '.jpg')
    f.close()

def draw_epochs(file):
    # 绘制每一个模型在不同层的欧式距离
    plt.figure()
    f = open('../filter1/filter_euclidean/' + file+'_ee1.txt', 'r')
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        y = []
        for i in range(len(line)):
            if i == 0 :continue
            y.append(float(line[i]))
        x = [str(i) for i in range(len(line)-1)]
        plt.plot(x, y, label=line[0])
        plt.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size': 8})
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '_ee1.jpg')
    f.close()

def draw_epochs1(file):
    # 绘制每一个模型在不同层的欧式距离之和
    plt.figure()
    f = open('../filter1/filter_euclidean/' + file+'_ee1.txt', 'r')
    f_new = open('../filter1/filter_euclidean_sk/' + file+'.txt', 'w')

    dicty = []
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        if len(dicty) == 0:
            dicty = [0 for i in range(len(line)-1)]
        for i in range(len(line)):
            if i == 0 :continue
            if dicty[i-1] == 0:
                dicty[i-1] = float(line[i])
            else:
                dicty[i - 1] += float(line[i])
    # x = [str(i) for i in range(len(dicty))]
    y = dicty
    k_function = []
    for i in range(len(dicty)):
        if i == 0: continue
        immediate = (dicty[i]-dicty[0])/i
        k_function.append(immediate)

    # 求正切de差值变化
    y = [k_function[0]]
    for i in range(len(k_function) - 1):
        immediate = abs(k_function[i] - k_function[i + 1])
        y.append(immediate)
    x = [str(i) for i in range(len(y))]
    # plt.plot(x, y)
    # plt.savefig('../filter1/filter_euclidean_pic/' + file + '_sumlayers.jpg')
    # 归一化操作
    y_norm = []
    miny = min(y)
    maxy = max(y)
    for i in y:
        y_norm.append((i - miny)/(maxy-miny))
    f_new.write(str(y_norm))
    # plt.plot(x, y_norm)
    # plt.savefig('../filter1/filter_euclidean_pic/' + file + '_sumlayersnorm.jpg')
    f.close()

def draw_epochs2(file):
    # 绘制epoch1的正切值求角度

    f = open('../filter1/filter_euclidean/' + file+'_ee1.txt', 'r')
    f_new = open('../filter1/filter_euclidean/' + file + '1k.txt', 'w')
    # 存ee1中tanΘ
    f_new1 = open('../filter1/filter_euclidean/' + file + '1kk.txt', 'w')
    #存Θ
    f_new2 = open('../filter1/filter_euclidean/' + file + '1kktheta.txt', 'w')
    # 存Θ差值
    f_new3 = open('../filter1/filter_euclidean/' + file + '1kktheta1.txt', 'w')

    dicty = []
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        if len(dicty) == 0:
            dicty = [0 for i in range(len(line)-1)]
        for i in range(len(line)):
            if i == 0 :continue
            if dicty[i-1] == 0:
                dicty[i-1] = float(line[i])
            else:
                dicty[i - 1] += float(line[i])
    k_function = []
    for i in range(len(dicty)):
        if i == 0: continue
        immediate = (dicty[i]-dicty[0])/i
        k_function.append(immediate)
        f_new.write(str(immediate)+'\n')
    x = [str(i + 1) for i in range(len(k_function))]
    plt.figure()
    plt.plot(x, k_function)
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '1k.jpg')
    plt.close()
    # 求正切de差值变化
    y = []
    for i in range(len(k_function)-1):
        immediate = abs(k_function[i]-k_function[i+1])
        y.append(immediate)
        f_new1.write(str(immediate)+'\n')
    x= [str(i+1) for i in range(len(y))]
    plt.figure()
    plt.plot(x, y)
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '1kk.jpg')
    plt.close()
    # 计算正切的jiaodu
    y_theta = []
    for i in range(len(k_function)):
        theta = math.degrees(math.atan(k_function[i]))
        y_theta.append(theta)
        f_new2.write(str(theta)+'\n')
    plt.figure()
    x = [str(i + 1) for i in range(len(y_theta))]
    plt.plot(x, y_theta)
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '1ktheta.jpg')

    y_theta1 = []
    for i in range(len(y_theta)-1):
        theta = abs(y_theta[i]-y_theta[i+1])
        y_theta1.append(theta)
        f_new3.write(str(theta) + '\n')
    x =  [str(i+1) for i in range(len(y_theta1))]
    plt.figure()
    plt.plot(x, y_theta1,'s')
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '1ktheta1.jpg')

    f.close()
    f_new.close()
    f_new1.close()
    f_new2.close()

def draw_epochs3(file):
    # 绘制epoch1的斜率求了一个差值
    plt.figure()
    f = open('../filter1/filter_euclidean/' + file+'_ee1.txt', 'r')
    f_new = open('../filter1/filter_euclidean/' + file + '_epoch3.txt', 'w')

    dicty = []
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        if len(dicty) == 0:
            dicty = [0 for i in range(len(line)-1)]
        for i in range(len(line)):
            if i == 0 :continue
            if dicty[i-1] == 0:
                dicty[i-1] = float(line[i])
            else:
                dicty[i - 1] += float(line[i])
    k_function = []
    for i in range(len(dicty)-1):
        if i == 0: continue
        immediate = abs(dicty[i]-dicty[i+1])
        k_function.append(immediate)
        f_new.write(str(immediate)+'\n')

    x= [str(i+1) for i in range(len(k_function))]
    plt.plot(x, k_function)
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '_e3.jpg')
    f.close()
    f_new.close()

def fund(x, a, b):
 return a*(np.log2(x))+b
def draw_epochs4(file):
    # 绘制epoch1的拟合曲线的斜率

    f = open('../filter1/filter_euclidean/' + file+'.txt', 'r')
    # 存储拟合曲线的斜率
    f_new = open('../filter1/filter_euclidean1/' + file + '_curvek.txt', 'w')

    dicty = []
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        if len(dicty) == 0:
            dicty = [0 for i in range(len(line)-1)]
        for i in range(len(line)):
            if i == 0 :continue
            if dicty[i-1] == 0:
                dicty[i-1] = float(line[i])
            else:
                dicty[i - 1] += float(line[i])
    y = dicty[1:]
    x= [i+1 for i in range(len(y))]
    # popt, pcov = curve_fit(fund, x,y)
    # y2 = [fund(i, popt[0], popt[1]) for i in x]
    z1 = np.polyfit(x, y, 5)
    p1 = np.poly1d(z1)
    y2 = p1(x)
    # 绘图
    x1 = [str(i) for i in x]
    plt.plot(x1, y, 's', label='original values')
    plt.plot(x1, y2, 'r', label='polyfit values')
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '_curve.jpg')
    plt.close()

    y = []
    for i in x:
        k = derivative(p1, i,dx=1e-6)
        y.append(k)
        f_new.write(str(k)+'\n')

    plt.plot(x1, y, 's', label='original values')
    plt.savefig('../filter1/filter_euclidean_pic/' + file + '_curvek.jpg')
    plt.close()

    f.close()
    f_new.close()

def diff(file):
    # 绘制epoch1的正切值求角度

    f = open('../filter1/filter_euclidean/' + file+'_ee1.txt', 'r')
    f_new = open('../filter1/filter_euclidean_skwess/' + file + '.txt', 'w')
    dicty = []
    for line_raw in f:
        line = line_raw.replace('\n','').split(' ')
        if len(dicty) == 0:
            dicty = [0 for i in range(len(line)-1)]
        for i in range(len(line)):
            if i == 0 :continue
            if dicty[i-1] == 0:
                dicty[i-1] = float(line[i])
            else:
                dicty[i - 1] += float(line[i])
    # k_function = []
    # for i in range(len(dicty)):
    #     if i == 0: continue
    #     immediate = (dicty[i]-dicty[0])/i
    #     k_function.append(immediate)
    #
    # # 求正切de差值变化
    # y = [k_function[0]]
    # for i in range(len(k_function) - 1):
    #     immediate = abs(k_function[i] - k_function[i + 1])
    #     y.append(immediate)
    # y_theta = []
    # for i in range(len(k_function)):
    #     theta = math.degrees(math.atan(k_function[i]))
    #     y_theta.append(theta)
    # y_theta1 = []
    # for i in range(len(y_theta) - 1):
    #     theta = abs(y_theta[i] - y_theta[i + 1])
    #     y_theta1.append(theta)

    y = []
    k_function = dicty
    for i in range(len(k_function) - 1):
        immediate = k_function[i+1] - k_function[i]
        y.append(immediate)

    for i in range(len(y)):
        if i < 3: continue
        f_new.write(str(i) + '\n')
        y_new = y[:i]
        print(str(i))
        #求方差
        a = 0
        b = 0
        start = 0
        vars = []

        while start !=len(y_new)-2:
            vars.append(sts.skewness(y_new[start:]))
            start += 1
        print(vars)

        length = len(vars)
        for i in range(length):
            if i == length:
                if vars[i] < 1:  a = i + 1
                else:b=''
            else:
                if vars[i] < 1 and vars[i + 1] < 1:
                    a = i + 1
                    break
                else:
                    a = ''

        end = 2
        vars0 = []
        while end !=len(y_new):
            y_end = y_new[:end]
            vars0.append(sts.skewness(y_end))
            end += 1
        print(vars0)

        for i in range(len(vars0)):
            if i ==len(vars0):
                if vars0[i] > 1 :
                    b = i+2
                else:
                    b = ''
            else:
                if vars0[i] > 1 and vars0[i+1] > 1:
                    b = i+2
                    break
                else:
                    b = ''

        print(file +' ['+str(a)+' , '+str(b)+']')

        f_new.write(str(vars)+'\n')
        f_new.write(str(vars0) + '\n')
        f_new.write(' ['+str(a)+' , '+str(b)+']'+'\n')

def main(argv=None):

    for file in os.listdir('../models'):
        if not os.path.isdir(file):
            try:
                epochs = 20
                # write_ee1(file, epochs)
                draw_epochs1(file)
            except IOError:
                print(file)
            except:
                print('epoch')



if __name__ == '__main__':
    main()