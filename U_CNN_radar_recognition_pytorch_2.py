# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:02:16 2018

@author: mingyang.wang
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.utils.data as data
from torch.autograd import Variable

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
    RF_data, PRI_data, PW_data, label = [], [], [], []
    files_name = os.listdir(path)
    global dir_dict2
    dir_dict = {}
    dir_dict2 = {}
    for file_name in files_name:
        num = file_name.split("_")
        if num[0] not in dir_dict:
            dir_dict[num[0]] = int(num[0])-1
            dir_dict2[int(num[0])-1] = num[0]
        file_path = os.path.join("%s\\%s" % (path, file_name))
        pdw_data = pd.read_csv(file_path, header=None, skiprows=[0], sep='\t')
        pdw_data = pdw_data.values 
        RF = encode_data(pdw_data[:, 0], 0, 10000, 7000)
        PRI = encode_data(pdw_data[:, 2], 0, 2000000, 7000)
        PW = encode_data(pdw_data[:,1], 0, 100000, 7000)
        RF_data.append(RF)
        PRI_data.append(PRI)
        PW_data.append(PW)
        label.append(dir_dict[num[0]])
    RF_data = np.array(RF_data).reshape(len(RF_data), 1, 7000).astype('float32')
    PRI_data = np.array(PRI_data).reshape(len(PRI_data), 1, 7000).astype('float32')
    PW_data = np.array(PW_data).reshape(len(PW_data), 1, 7000).astype('float32')
    label = np.array(label).astype('int64')
    #label = make_one_hot(label, len(dir_dict)).astype('int64')
    return RF_data, PRI_data, PW_data, label

def data_iter(x, y, z, l, batch_size):
    for i in range(0, 560, batch_size):
        yield (x[i:i+batch_size], y[i:i+batch_size], z[i:i+batch_size], l[i:i+batch_size])


class U_CNN(nn.Module):
    def __init__(self):
        super(U_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 7)
        self.conv2 = nn.Conv1d(10, 15, 5)
        self.conv3 = nn.Conv1d(15, 20, 3)
        self.pool = nn.MaxPool1d(4,4)
        self.fc1 = nn.Linear(20*108*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 8)
        
    def forward(self, x, y, z):
        #x = self.conv1(x)
        #x = F.relu(x)
        #x = self.pool(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.item)
        x = x.view(-1, 20*108)
        y = self.pool(F.relu(self.conv1(y)))
        y = self.pool(F.relu(self.conv2(y)))
        y = self.pool(F.relu(self.conv3(y)))
        y = y.view(-1, 20*108)
        z = self.pool(F.relu(self.conv1(z)))
        z = self.pool(F.relu(self.conv2(z)))
        z = self.pool(F.relu(self.conv3(z)))
        z = z.view(-1, 20*108)
        u = torch.cat((x, y, z), 1)
        u = F.relu(self.fc1(u))
        u = F.relu(self.fc2(u))
        u = F.relu(self.fc3(u))
        output = self.fc4(u)
        return output
        
PATH = 'E:\\data\\PRI\\1031\\train'
RF_data, PRI_data, PW_data, label = load_data(PATH)
#train_dataset = data.TensorDataset((RF_data, PRI_data, PW_data), label)
#train_loader = data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

ucnn = U_CNN()
ucnn.load_state_dict(torch.load('E:\\model_save\\UCNN_weights_1105_pt.pkl')) 
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ucnn.parameters(), lr=0.002, momentum=0.9)
for epoch in range(100):
    running_loss = 0
    for i, data in enumerate(data_iter(RF_data, PRI_data, PW_data, label, 20)):
        x, y, z, l = torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])
        #print(x.size())
        #x.double()
        #print(x)
        x, y, z, l = Variable(x), Variable(y), Variable(z), Variable(l)
        optimizer.zero_grad()
        outputs = ucnn(x, y, z)
        #print(outputs.item())
        loss = criterion(outputs, l)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 28 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 28))
            running_loss = 0.0

print('Finished Training')
        
torch.save(ucnn.state_dict(), 'E:\\model_save\\UCNN_weights_1105_pt.pkl')        
"""
#预测
ucnn.load_state_dict(torch.load('E:\\model_save\\UCNN_weights_1105_pt.pkl')) 
test_dir_path = "E:\\data\\PRI\\1105\\test_200"
RF_data, PRI_data, PW_data, label = load_data(test_dir_path)
x, y, z = torch.from_numpy(RF_data), torch.from_numpy(PRI_data), torch.from_numpy(PW_data)
test_output = ucnn(x, y, z)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
acc = 0
for i in zip(pred_y, label):
    if i[0] == i[1]:
        acc += 1
    else:
        print(i[1], '-->', i[0])
        
print('Accuracy:', acc/len(label))
#print(pred_y)
#print(label)
