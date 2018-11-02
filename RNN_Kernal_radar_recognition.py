# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:34:08 2018

@author: mingyang.wang
"""

import os
import torch
import numpy as np
import pandas as pd
#import torch.utils.data as data
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD

#将标签转换成独热编码格式
def make_one_hot(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)

#加载数据
def load_data(path):
    data, label = [], []
    file_names = os.listdir(path)
    #gaus = GaussianRandomProjection(n_components=10, random_state=1)
    for file_name in file_names:
        fl = file_name.split('_')
        if fl[0] not in label2index:
            label2index[fl[0]] = int(fl[0])-1
            index2label[int(fl[0])-1] = fl[0]    
        file_path = os.path.join('%s//%s' % (path, file_name))
        df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None)
        #df_guas = gaus.fit_transform(df.values)
        #data.append(df_guas)
        data.append(df.values)
        label.append(label2index[fl[0]])
    data = np.array(data)
    label = make_one_hot(np.array(label), len(set(label)))
    return data, label

global label2index, index2label
label2index, index2label = {}, {}
PATH = "E:\\data\\PRI\\1031\\train"
data, label = load_data(PATH)
print(data.shape)

"""
train_data = torch.from_numpy(data1)
train_label = torch.from_numpy(label)
train_dataset = data.TensorDataset(train_data, train_label)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

#std = StandardScaler()
#data_std = std.fit_transform(data)
#gaus = GaussianRandomProjection(n_components=[100,10], random_state=1)
#data_guas = gaus.fit_transform(data)
#print(data.shape)

class RNN(torch.nn.Module):
    def __init__(self):
        super (RNN, self).__init__()
        self.gru = torch.nn.GRU(10, 512, 2)
        
    def forward(self, input,):
        output, hidden = self.gru(input)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


rnn = RNN()
optimozer = torch.optim.Adam(rnn.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()

loss_list = []
flg_loss = 0
for epoch in range(10):
    print("epoch:{}, flg_loss:{}".format(epoch, flg_loss))
    for i,(x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        print(batch_x.shape)
        output, hidden = rnn(batch_x)
        loss = loss_func(output, batch_y)
        flg_loss = loss.detach().item()
        loss_list.append(loss.detach().item())
        optimozer.zero_grad()
        loss.backward()
        optimozer.step()
"""

model = Sequential()
model.add(LSTM(16,return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(8, activation=("softmax")))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd)

#model.load_weights('E:\\model_save\\kernal_rnn_radar_demo_1102.h5')

model.fit(data, label, batch_size=50, epochs=20)
model.save_weights('E:\\model_save\\kernal_rnn_radar_demo_1102.h5')