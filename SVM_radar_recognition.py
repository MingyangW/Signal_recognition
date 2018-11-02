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
from sklearn.random_projection import GaussianRandomProjection




#将标签转换成独热编码格式
def make_one_hot(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)


#加载数据
def load_data(path):
    df = pd.read_csv(path, sep='\t',skiprows=0, encoding='utf-8')
    df = pd.get_dummies(df, columns=["RF类型", "PW类型", "PRI类型"]).astype('int')
    data = np.array(df.iloc[:, 2:])
    label = np.array(df.iloc[:, 1])
    return data, label

PATH = "E:\\data\\PRI\\1032\\train.txt"
data, label = load_data(PATH)
PATH1 = "E:\\data\\PRI\\1032\\test.txt"
data_test, label_test = load_data(PATH1)
std = StandardScaler()
data_std = std.fit_transform(data)
data_test_std = std.transform(data_test)



"""
gauss = GaussianRandomProjection(n_components=10, random_state=1)
#data_gauss = gauss.fit_transform(data[3:4])  
data_test_gauss = gauss.fit_transform(data_test)
print(data_test_gauss[:3])
#print(data_gauss.shape)

grid = GridSearchCV(svm.SVC(), param_grid={'C':[0.1, 1, 20, 15], 'gamma':[10,0.1, 1, 0.01]}, cv=4)
grid.fit(data_std, label)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


"""
clf = svm.SVC(C=10, gamma=1)
clf.fit(data_std, label)
prediction = clf.predict(data_test_std)

acc = accuracy_score(label_test, prediction)
print(acc)
for i in range(len(label_test)):
    if label_test[i] != prediction[i]:
        print(label_test[i], "-->", prediction[i])
        print(i)
        print(data_test[i])

