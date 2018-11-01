# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:54:56 2018

@author: mingyang.wang
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


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

#加载数据
def load_data(path):
    df = pd.read_csv(path, sep='\t',skiprows=1, encoding='utf-8')
    data = np.array(df.iloc[:, 2:])
    label = np.array(df.iloc[:, 1])-1
    #label = make_one_hot(label, len(set(label)))
    #print(data[:5])
    #print(label[:5])
    return data, label

PATH = "E:\\data\\PRI\\1032\\train.txt"
data, label = load_data(PATH)
PATH1 = "E:\\data\\PRI\\1032\\test.txt"
data_test, label_test = load_data(PATH1)
std = StandardScaler()
data_std = std.fit_transform(data)
data_test_std = std.transform(data_test)
"""
grid = GridSearchCV(svm.SVC(), param_grid={'C':[0.1, 1, 20, 15], 'gamma':[10,0.1, 1, 0.01]}, cv=4)
grid.fit(data_std, label)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


"""
clf = svm.SVC(C=10, gamma=1)
clf.fit(data_std, label)
prediction = clf.predict(data_test_std)


#predict = np.argmax(prediction)
print(prediction)
print(prediction.shape)
print(label_test.shape)   
acc = accuracy_score(label_test, prediction)
print(acc)

