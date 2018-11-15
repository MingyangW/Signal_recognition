# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:37:17 2018

@author: mingyang.wang
"""


import os
import argparse
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, pooling, Concatenate, Flatten


# --将标签转换成独热编码格式
def make_one_hot(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)

# --将将样本特征进行编码
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

# --通过TOA计算PRI
def compute_diff(toa, n):
    toa_a = toa[n:]
    toa_b = toa[:(toa.shape[0]-n)]
    return toa_a - toa_b
        

# --加载数据
def load_data(class2index, opt):
    RF_data, PRI_data, PW_data, label = [], [], [], []
    try:
        files_name = os.listdir(opt.data_path) if os.path.exists(opt.data_path) else print('File is not exist!')
        for file_name in files_name:
            file_path = os.path.join("%s\\%s" % (opt.data_path, file_name))
            pdw_data = pd.read_csv(file_path, header=None, skiprows=[0], sep='\t')
            pdw_data = pdw_data.values 
            RF_data.append(encode_data(pdw_data[:, 0], 0, 10000, 7000))
            PRI_data.append(encode_data(pdw_data[:, 2], 0, 2000000, 6000))
            PW_data.append(encode_data(pdw_data[:,1], 0, 100000, 8000))
            if opt.function != 'predict':
                num = file_name.split("_")
                label.append(class2index[num[0]]) if num[0] in class2index else print('NO this class!')
            else:
                label.append(file_name[:-4])
        RF_data = np.array(RF_data).reshape(len(RF_data), 7000, 1)
        PRI_data = np.array(PRI_data).reshape(len(PRI_data), 6000, 1)
        PW_data = np.array(PW_data).reshape(len(PW_data), 8000, 1)
        if opt.function != 'predict':
            label = make_one_hot(np.array(label), len(class2index))
        return RF_data, PRI_data, PW_data, label
    except:
        print('Load_data error!')

#U_CNN模型
def ucnn_model(RF_data, PRI_data, PW_data, label, class2index, opt ):
    
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
    xyz = Dense(512, activation="relu", name='dense_1')(xyz)
    xyz = Dense(256, activation="relu", name='dense_2')(xyz)
    xyz = Dense(128, activation="relu", name='dense_3')(xyz)
    output = Dense(opt.n_classes, activation="softmax", name='dense_4')(xyz)

    model = Model(inputs=[RF_input, PRI_input, PW_input], outputs=output)

    path = 'E:\\model_save\\Signal_recognition\\UCNN_weights_{}_{}.h5'.format(opt.n_classes, str(opt.train_data_num))
    if opt.function == 'train':
        if os.path.exists(path):
            model.load_weights(path, by_name=True)
        sgd = SGD(lr=opt.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit({"RF_input":RF_data, "PRI_input":PRI_data, "PW_input":PW_data},\
                  label, batch_size=opt.batch_size, epochs=opt.epochs)
        model.save_weights(path) 
    else:
        if os.path.exists(path):
            model.load_weights(path)
            prediction = model.predict({"RF_input":RF_data, "PRI_input":PRI_data, "PW_input":PW_data})
            label_num = np.argmax(prediction, axis=1)
            
            if opt.function == 'test':
                acc_num = 0
                label_true = np.argmax(label, axis=1)
                for i in range(len(label_num)):
                    if label_true[i] == label_num[i]:
                        acc_num += 1
                    else:
                        print(label_true[i], "——>", label_num[i])
                print(acc_num/len(label_true))
            else:
                print(label_num)
                if opt.save_predict:
                    label = np.array(label).reshape(-1, 1)  
                    index2class = {item[1]:item[0] for item in class2index.items()}
                    predict_class = list(map(lambda x:index2class[x], label_num))
                    predict_class = np.array(predict_class).reshape(-1, 1)
                    predict = np.concatenate((label, predict_class), axis=1)
                    df = pd.DataFrame(predict, columns=['file_name', 'class'])
                    df.to_csv('E:\\data\\signal_recognition\\predict\\predict.txt', sep='\t', index=False)
        else:
            print('No model!')
        

def main():
    path = 'E:\\data\\signal_recognition\\test_25'
    parser = argparse.ArgumentParser()
    parser.add_argument('-function', type=str, choices=['train', 'test', 'predict'], default='test')
    parser.add_argument('-data_path', type=str,  default=path)#required=True,
    parser.add_argument('-n_classes', type=int, default=8) #--分类数量
    parser.add_argument('-train_data_num', type=int, default=100) #--训练样本数量
    #parser.add_argument('-test_data_num', type=int, default=25)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-save_predict', action='store_true', default=True) #--将预测结果保存文件

    class2index = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7}
    
    opt = parser.parse_args()
    RF_data, PRI_data, PW_data, label = load_data(class2index, opt)
    ucnn_model(RF_data, PRI_data, PW_data, label, class2index, opt)

if __name__ == '__main__':
    main()