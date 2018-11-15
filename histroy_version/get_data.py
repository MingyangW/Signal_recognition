# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:03:27 2018

@author: mingyang.wang
"""

#import random
import numpy as np
import pandas as pd

def normal_scope(m, lam):
    while True:
        a = np.random.normal(m, lam)
        if -3*lam <= a <= 3*lam:
            return int(a)

def RF_fix(cent, scope, num):
    output = []
    while len(output)<num:
        output.append(cent + normal_scope(0, scope))
    return output

def RF_agile(minrf, maxrf, num):
    output = [np.random.randint(minrf, maxrf)]
    while len(output)<num:
        a = np.random.randint(minrf, maxrf)
        for i in output:
            if abs(a-i)<10:
                break
        else:
            output.append(a)
    return output
        
def PW_vari(scope, num):
    output = []
    while len(output)<num:
        a  = np.random.choice(scope)
        output.append(a)
        scope.remove(a)
    return sorted(output)
        
def PRI_jit(cent, scope):
    return cent + np.random.randint(-scope, scope)

def PRI_st(scope):
    while True:
        pri = np.random.choice(scope)
        output = [pri]
        a = [0.02, 0.04, 0.08]
        for i in a:
            k = int(output[-1]*(1+i))
            if k < max(scope):
                output.append((k//1000)*1000)
            else:
                break
        else:
            return output


"""
x = []
y = []
for i in range(20):
    x.append(np.random.randint(0, 4))
    y.append(int(np.random.normal(1000, 2)))

    
a = np.random.randint(-30, 30)
RF1 = [9127, 9375, 9410, 9440, 9490]
RF3 = [i for i in range(8500, 9601, 50)]
print(RF3)
RF = np.random.choice(RF1)
RF = 9375 + normal_scope(0, 10)
RF = RF_agile(8500, 9600, 4)
print(RF)

PW1 = [180, 300, 600, 1200]
PW2 = [i for i in range(150, 1501, 150)]
pw = PW1[np.random.randint(0, 3)]
pw = PW_vari(PW2, 4)

PRI1 = {1000:50, 2000:100}
PRI = PRI_jit(1000, 50)
c = np.random.choice(PW1)
random.shuffle(PW1)
print(PW1)
"""
global data_para
data_para = []    
        
def get_para1():
    PW1 = [i for i in range(50, 1001, 50)]
    PRI1 = [i for i in range(125000, 2500001, 1000)]
    rf = np.random.choice([9127, 9375, 9410, 9440, 9490])
    pw = PW_vari(PW1, 4)
    pri = PRI_st(PRI1)
    return rf, pw, pri

def get_data1(num):
    rf_p, pw_p, pri_p = get_para1()
    global data_para
    data_para.append([num,1, 1, rf_p, 0, 0, 0, 2, pw_p[0], pw_p[1], pw_p[2], pw_p[3], 3, \
                     pri_p[0], pri_p[1], pri_p[2], pri_p[3]])
    rf, pw, pri = np.zeros(1000),np.zeros(1000),np.zeros(1000)
    rf[:] = rf_p
    pw[:] = 250*pw_p
    pri[:] = 250*pri_p
    rf = rf.reshape(-1,1)
    pw = pw.reshape(-1,1)
    pri = pri.reshape(-1,1)
    data = np.concatenate([rf,pw,pri], axis=1)
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1030\\train\\1_{}.txt".format(num)
    df.to_csv(path, sep="\t", index=False)


def get_para2():
    PW1 = [60, 600]
    PRI1 = [250000, 500000]
    rf = np.random.choice([i for i in range(9345, 9406)])
    pw = np.random.choice(PW1)
    pri = np.random.choice(PRI1)
    return rf, pw, pri

def get_data2(num):
    rf_p, pw_p, pri_p = get_para2()
    global data_para
    data_para.append([num,2, 1, rf_p, 0, 0, 0, 1, pw_p, 0, 0,0, 2, \
                     int(pri_p*0.95), int(pri_p*1.05), 0, 0])
    #print(rf_p, pw_p, pri_p)
    rf, pw, pri = np.zeros(1000),np.zeros(1000),np.zeros(1000)
    rf[:] = rf_p
    pw[:] = pw_p
    pri[:] = [PRI_jit(pri_p, int(pri_p*0.05)) for i in range(1000)]
    rf = rf.reshape(-1,1)
    pw = pw.reshape(-1,1)
    pri = pri.reshape(-1,1)
    data = np.concatenate([rf,pw,pri], axis=1)
    #print(data[:2])
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1030\\train\\2_{}.txt".format(num)
    df.to_csv(path, sep="\t", index=False)


def get_para3():
    PW1 = [500, 600]
    PRI1 = [1250000, 800000, 500000]
    rf = np.random.choice([i for i in range(9345, 9406)])
    pw = np.random.choice(PW1)
    pri = np.random.choice(PRI1)
    return rf, pw, pri

def get_data3(num):
    rf_p, pw_p, pri_p = get_para3()
    global data_para
    data_para.append([num,3, 1, rf_p, 0, 0, 0, 1, pw_p, 0, 0,0, 1, \
                     pri_p, 0, 0, 0])
    #print(rf_p, pw_p, pri_p)
    rf, pw, pri = np.zeros(1000),np.zeros(1000),np.zeros(1000)
    rf[:] = rf_p
    pw[:] = pw_p
    pri[:] = pri_p
    rf = rf.reshape(-1,1)
    pw = pw.reshape(-1,1)
    pri = pri.reshape(-1,1)
    data = np.concatenate([rf,pw,pri], axis=1)
    #print(data[:2])
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1030\\train\\3_{}.txt".format(num)
    df.to_csv(path, sep="\t", index=False)


def get_para4():
    PW1 = [i for i in range(150, 1501, 50)]
    PRI1 = [i for i in range(333000, 4000001, 1000)]
    rf = RF_agile(8500, 9600, 4)
    pw = PW_vari(PW1,4)
    pri = PRI_st(PRI1)
    return rf, pw, pri

def get_data4(num):
    rf_p, pw_p, pri_p = get_para4()
    global data_para
    data_para.append([num,4, 2, rf_p[0], rf_p[1], rf_p[2], rf_p[3], 2, pw_p[0],\
                     pw_p[1], pw_p[2],pw_p[3], 3, \
                     pri_p[0], pri_p[1], pri_p[2], pri_p[3]])
    #print(rf_p, pw_p, pri_p)
    rf, pw, pri = np.zeros(1000),np.zeros(1000),np.zeros(1000)
    rf[:] = 250*rf_p
    pw[:] = 250*pw_p
    pri[:] = 250*pri_p
    rf = rf.reshape(-1,1)
    pw = pw.reshape(-1,1)
    pri = pri.reshape(-1,1)
    data = np.concatenate([rf,pw,pri], axis=1)
    #print(data[:2])
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1030\\train\\4_{}.txt".format(num)
    df.to_csv(path, sep="\t", index=False)


def get_para5():
    PW1 = [1000, 600]
    PRI1 = [1000000, 500000]
    rf = RF_agile(8500, 9600, 4)
    pw = np.random.choice(PW1)
    pri = np.random.choice(PRI1)
    return rf, pw, pri

def get_data5(num):
    rf_p, pw_p, pri_p = get_para5()
    global data_para
    data_para.append([num,5, 2, rf_p[0], rf_p[1], rf_p[2], rf_p[3], 1, pw_p, 0, 0,0, 1, \
                     pri_p, 0, 0, 0])
    #print(rf_p, pw_p, pri_p)
    rf, pw, pri = np.zeros(1000),np.zeros(1000),np.zeros(1000)
    rf[:] = 250*rf_p
    pw[:] = pw_p
    pri[:] = pri_p
    rf = rf.reshape(-1,1)
    pw = pw.reshape(-1,1)
    pri = pri.reshape(-1,1)
    data = np.concatenate([rf,pw,pri], axis=1)
    #print(data[:2])
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1030\\train\\5_{}.txt".format(num)
    df.to_csv(path, sep="\t", index=False)



for i in range(25):
    if i < 5:
        get_data1(i)
    elif 5<= i<10:
        get_data2(i)
    elif 10<= i<15:
        get_data3(i)
    elif 15<= i<20:
        get_data4(i)
    else:
        get_data5(i)
    df = pd.DataFrame(
            data_para, columns=["序号","型号","RF类型","RF1","RF2","RF3","RF4","PW类型",\
                                "PW1","PW2","PW3","PW4","PRI类型","PRI1","PRI2",\
                                "PRI3","PRI4"])
    df.to_csv("E:\\data\\PRI\\1030\\train.txt", sep="\t")


global RF_dict, PW_dict, PRI_dict
RF_dict = {1:'fix', 2:'agile'}
PW_dict = {1:'fix', 2:'variation'}
PRI_dict = {1:'fix', 2:'jitter', 3:'stagger'}
def get_data(order, radar_class, rf_type, rf, pw_type, pw, pri_type, pri, \
             jitter=None, rf_num=None, pw_num=None, pri_num=None):
    if RF_dict[rf_type] == 'fix':
        rf_p = np.random.choice(rf)
        rf_l = [1, rf_p, 0, 0, 0]
    elif RF_dict[rf_type] == 'agile':
        rf_p = RF_agile(rf[0], rf[1], 4)
        rf_l = [2, rf_p[0], rf_p[1], rf_p[2], rf_p[3]]
    else:
        print("RF No this type!")
    if PW_dict[pw_type] == 'fix':
        pw_p = np.random.choice(pw)
        pw_l = [1, pw_p, 0, 0, 0]
    elif PW_dict[pw_type] == 'variation':
        pw_p = PW_vari(pw, 4)
        pw_l = [2, pw_p[0], pw_p[1], pw_p[2], pw_p[3]]
    else:
        print('PW NO this type!')
    if PRI_dict[pri_type] == 'fix':
        pri_p = np.random.choice(pri)
        pri_l = [1, pri_p, 0, 0, 0]
    elif PRI_dict[pri_type] == 'jitter':
        pri_p = np.random.choice(pri)
        pri_l = [2, pri_p, 0, 0, 0]
    elif PRI_dict[pri_type] == 'stagger':
        pri_p = PRI_st(pri)
    else:
        print('PRI NO this type!')