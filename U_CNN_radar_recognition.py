# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:26:40 2018

@author: mingyang.wang
"""

import os
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, pooling, Concatenate, Flatten

global class2index, index2class
class2index, index2class = {}, {}

#将标签转换成独热编码格式
def make_one_hot(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)

#将将样本特征进行编码
def encode_data(data, binmin, binmax, binnum):
    b = (binmax - binmin) / binnum
    output = [0 for i in range(binnum)]
    for i in range(len(data)):
        if binmin < data[i] < binmax:
            output[int((data[i]-binmin)//b)] = 1
        elif data[i] <= binmin:
            output[0] = 1
        else:
            output[-1] = 1
    return output

#通过TOA计算PRI
def compute_diff(toa, n):
    toa_a = toa[n:]
    toa_b = toa[:(toa.shape[0]-n)]
    return toa_a - toa_b
        

#加载数据
def load_data(path):
    global dir_dict2
    dir_dict = {}
    dir_dict2 = {}
    RF_data, PRI_data, PW_data, label = [], [], [], []
    files_name = os.listdir(path)

    for file_name in files_name:
        num = file_name.split("_")
        if num[0] not in dir_dict:
            dir_dict[num[0]] = int(num[0])-1
            dir_dict2[int(num[0])-1] = num[0]
        file_path = os.path.join("%s\\%s" % (path, file_name))
        pdw_data = pd.read_csv(file_path, header=None, skiprows=[0], sep='\t')
        pdw_data = pdw_data.values 
        RF = encode_data(pdw_data[:, 0], 0, 10000, 7000)
        #diff = compute_diff(pdw_data[:,], 1)
        PRI = encode_data(pdw_data[:, 2], 0, 2000000, 6000)
        PW = encode_data(pdw_data[:,1], 0, 100000, 8000)
        RF_data.append(RF)
        PRI_data.append(PRI)
        PW_data.append(PW)
        label.append(dir_dict[num[0]])
    RF_data = np.array(RF_data).reshape(len(RF_data), 7000, 1)
    PRI_data = np.array(PRI_data).reshape(len(PRI_data), 6000, 1)
    PW_data = np.array(PW_data).reshape(len(PW_data), 8000, 1)
    label = np.array(label)
    label = make_one_hot(label, len(dir_dict))
    return RF_data, PRI_data, PW_data, label

#U_CNN模型
def ucnn_model(RF_data, PRI_data, PW_data, label=None, model_type='test'):
    RF_input = Input(shape=(7000, 1), name="RF_input")
    PRI_input = Input(shape=(6000, 1), name="PRI_input")
    PW_input = Input(shape=(8000, 1), name="PW_input")

    x = Conv1D(10, 7, activation="relu", name='conv_x1')(RF_input)
    x = pooling.MaxPooling1D(4, 4)(x)
    x = Conv1D(15, 5, activation="relu", name='conv_x2')(x)
    x = pooling.MaxPooling1D(4, 4)(x)
    x = Conv1D(20, 3, activation="relu", name='conv_x3')(x)
    x = pooling.MaxPooling1D(4, 4)(x)
    x = Flatten()(x)

    y = Conv1D(10, 7, activation="relu", name='conv_y1')(PRI_input)
    y = pooling.MaxPooling1D(4, 4)(y)
    y = Conv1D(15, 5, activation="relu", name='conv_y2')(y)
    y = pooling.MaxPooling1D(4, 4)(y)
    y = Conv1D(20, 3, activation="relu", name='conv_y3')(y)
    y = pooling.MaxPooling1D(4, 4)(y)
    y = Flatten()(y)

    z = Conv1D(10, 7, activation="relu", name='conv_z1')(PW_input)
    z = pooling.MaxPooling1D(4, 4)(z)
    z = Conv1D(15, 5, activation="relu", name='conv_z2')(z)
    z = pooling.MaxPooling1D(4, 4)(z)
    z = Conv1D(20, 3, activation="relu", name='conv_z3')(z)
    z = pooling.MaxPooling1D(4, 4)(z)
    z = Flatten()(z)

    xyz = Concatenate(axis=1)([x, y, z])
    #xyz = Dense(1024, activation="relu", name='dense_0')(xyz)
    xyz = Dense(512, activation="relu", name='dense_1')(xyz)
    xyz = Dense(256, activation="relu", name='dense_2')(xyz)
    xyz = Dense(128, activation="relu", name='dense_3')(xyz)
    output = Dense(8, activation="softmax", name='dense_4')(xyz)

    model = Model(inputs=[RF_input, PRI_input, PW_input], outputs=output)
    #print(model.summary()) 
    if model_type == 'train':
        model.load_weights('E:\\model_save\\UCNN_weight\\UCNN_weights_1101_100.h5', by_name=True)#, by_name=True
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit({"RF_input":RF_data, "PRI_input":PRI_data, "PW_input":PW_data}, label, batch_size=20, epochs=20)
        model.save_weights('E:\\model_save\\UCNN_weight\\UCNN_weights_1101_100.h5') 
    elif model_type == 'test':
        model.load_weights('E:\\model_save\\UCNN_weight\\UCNN_weights_1101_100.h5')
        prediction = model.predict({"RF_input":RF_data, "PRI_input":PRI_data, "PW_input":PW_data})
        #print(prediction)
        lable_num = np.argmax(prediction, axis=1)
        #print(np.argmax(prediction, axis=1))
        label_true = np.argmax(label, axis=1)
        acc_num = 0
        global dir_dict2
        for i in range(len(lable_num)):
            if label_true[i] == lable_num[i]:
                acc_num += 1
            else:
                print(label_true[i], "——>", lable_num[i])
            
            #print(dir_dict2[lable_num[i]])
        print(acc_num/len(label_true))
    else:
        print("Don't have this type!")

global dir_dict2


def main():
    #PATH = "E:\\data\\PRI\\1031\\train"
    PATH = "E:\\data\\PRI\\1105\\test_200"
    RF_data, PRI_data, PW_data, label = load_data(PATH)
    print(RF_data.shape, PRI_data.shape, PW_data.shape, label.shape)
    #ucnn_model(RF_data, PRI_data, PW_data, label, model_type='train')
    ucnn_model(RF_data, PRI_data, PW_data, label, model_type='test')

if __name__ == '__main__':
    main()