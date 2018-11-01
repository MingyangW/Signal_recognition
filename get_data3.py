# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:44:45 2018

@author: mingyang.wang
"""

import numpy as np
import pandas as pd


def RF_agile(scope, num):
    output = [np.random.choice(scope)]
    while len(output)<num:
        a = np.random.choice(scope)
        for i in output:
            if abs(a-i)<10:
                break
        else:
            output.append(a)
    return sorted(output)
        
def PW_vari(scope1, num):
    scope = scope1.copy()
    output = []
    while len(output)<num:
        a  = np.random.choice(scope)
        output.append(a)
        scope.remove(a)
    return sorted(output)

def PRI_jit(cent, scope):
    return cent + np.random.randint(-scope, scope)

def PRI_st(scope, stagger):
    while True:
        pri = np.random.choice(scope)
        output = [pri]
        for i in stagger:
            k = int(output[-1]*(1+i))
            if k < max(scope):
                output.append((k//1000)*1000)
            else:
                break
        else:
            return output

global RF_dict, PW_dict, PRI_dict, data_para
RF_dict = {1:'fix', 2:'agile'}
PW_dict = {1:'fix', 2:'variation'}
PRI_dict = {1:'fix', 2:'jitter', 3:'stagger'}
data_para = []
def get_data(order, radar_class, rf_type, rf, pw_type, pw, pri_type, pri, \
             jitter=None, stagger=None, rf_num=4, pw_num=4, pri_num=4):
    num = 1000
    rf_d, pw_d, pri_d = np.zeros(num),np.zeros(num),np.zeros(num)
    if RF_dict[rf_type] == 'fix':
        rf_p = np.random.choice(rf)
        rf_l = [1, rf_p, 0, 0, 0]
        rf_d[:] = rf_p
    elif RF_dict[rf_type] == 'agile':
        rf_p = RF_agile(rf, rf_num)
        rf_l = [2, rf_p[0], rf_p[1], rf_p[2], rf_p[3]]
        rf_d[:] = (num*rf_p)[:num]
    else:
        print("RF No this type!")
    if PW_dict[pw_type] == 'fix':
        pw_p = np.random.choice(pw)
        pw_l = [1, pw_p, 0, 0, 0]
        pw_d[:] = pw_p
    elif PW_dict[pw_type] == 'variation':
        pw_p = PW_vari(pw, pw_num)
        pw_l = [2, pw_p[0], pw_p[1], pw_p[2], pw_p[3]]
        pw_d[:] = (num*pw_p)[:num]
    else:
        print('PW NO this type!')
    if PRI_dict[pri_type] == 'fix':
        pri_p = np.random.choice(pri)
        pri_l = [1, pri_p, 0, 0, 0]
        pri_d[:] = pri_p
    elif PRI_dict[pri_type] == 'jitter':
        pri_p = np.random.choice(pri)
        pri_d[:] = [PRI_jit(pri_p, int(pri_p*jitter)) for i in range(num)]
        pri_l = [2, int(min(pri_d)), int(max(pri_d)), 0, 0]
    elif PRI_dict[pri_type] == 'stagger':
        pri_p = PRI_st(pri, stagger)
        pri_l = [3, pri_p[0], pri_p[1], pri_p[2], pri_p[3]]
        pri_d[:] = (num*pri_p)[:num]
    else:
        print('PRI NO this type!')
    data_para.append([order, radar_class] + rf_l + pw_l + pri_l)
    rf_d = rf_d.reshape(-1,1)
    pw_d = pw_d.reshape(-1,1)
    pri_d = pri_d.reshape(-1,1)
    data = np.concatenate([rf_d, pw_d, pri_d], axis=1)
    df = pd.DataFrame(data, columns=["rf", "pw", "pri"])
    path = "E:\\data\\PRI\\1101\\train_20\\{}_{}.txt".format(radar_class, order)
    df.to_csv(path, sep="\t", index=False)
    
def scope_list(minv, maxv, step=1):
    return [i for i in range(minv, maxv, step)]

rf1 = scope_list(9345, 9406)
rf2 = scope_list(9315, 9376)
rf3 = scope_list(8500, 9600)
rf4 = scope_list(8700, 9400)
rf6 = scope_list(9315, 9376)
rf7 = scope_list(9285, 9346)
rf8 = scope_list(9255, 9316)

pw1 = [180, 300, 600, 1200]
pw2 = [150, 300, 600, 1000]
pw3 = scope_list(150, 1501, step=50)
pw4 = scope_list(100, 1001, step=50)
pw6 = [100, 500, 800]
pw7 = [150, 300, 700]
pw8 = [100, 200, 600]

pri1, jitter1 = [250000, 500000], 0.05
pri2, jitter2 = [300000, 600000], 0.07
pri3, stagger3 = scope_list(200000, 600001, 1000), [0.02, 0.04, 0.08]
pri4, stagger4 = scope_list(100000, 500001, 1000), [0.03, 0.06, 0.012]
pri6, jitter6 = [200000, 300000, 400000], 0.05
pri7, jitter7 = [220000, 330000, 450000], 0.10
pri8, jitter8 = [200000, 300000, 500000], 0.07


N = 560
for i in range(N):
    """
    if i < N/16: 
        get_data(i, 1, 1, rf1, 1, pw1, 2, pri1, jitter=jitter1)
    elif N/16 <= i < N/8:
        get_data(i, 2, 1, rf2, 1, pw2, 2, pri2, jitter=jitter2)
    elif N/16*2 <= i < N/16*3:
        get_data(i, 3, 1, rf1, 1, pw1, 3, pri3, stagger=stagger3)
    elif N/16*3 <= i < N/16*4:
        get_data(i, 4, 1, rf2, 1, pw2, 3, pri4, stagger=stagger4)
    elif N/16*4 <= i < N/16*5:
        get_data(i, 5, 1, rf1, 2, pw3, 2, pri1, jitter=jitter1)
    elif N/16*5 <= i < N/16*6:
        get_data(i, 6, 1, rf2, 2, pw4, 2, pri2, jitter=jitter2)
    elif N/16*6 <= i < N/16*7:
        get_data(i, 7, 1, rf1, 2, pw3, 3, pri3, stagger=stagger3)
    elif N/16*7 <= i < N/16*8:
        get_data(i, 8, 1, rf2, 2, pw4, 3, pri4, stagger=stagger4)
    elif N/16*8 <= i < N/16*9:
        get_data(i, 9, 2, rf3, 1, pw1, 2, pri1, jitter=jitter1)
    elif N/16*9 <= i < N/16*10:
        get_data(i, 10, 2, rf4, 1, pw2, 2, pri2, jitter=jitter2)
    elif N/16*10 <= i < N/16*11:
        get_data(i, 11, 2, rf3, 1, pw1, 3, pri3, stagger=stagger3)
    elif N/16*11 <= i < N/16*12:
        get_data(i, 12, 2, rf4, 1, pw2, 3, pri4, stagger=stagger4)
    elif N/16*12 <= i < N/16*13:
        get_data(i, 13, 2, rf3, 2, pw3, 2, pri1, jitter=jitter1)
    elif N/16*13 <= i < N/16*14:
        get_data(i, 14, 2, rf4, 2, pw4, 2, pri2, jitter=jitter2)
    elif N/16*14 <= i < N/16*15:
        get_data(i, 15, 2, rf3, 2, pw3, 3, pri3, stagger=stagger3)
    else :
        get_data(i, 16, 2, rf4, 2, pw4, 3, pri4, stagger=stagger4)
    """
    if 500<= i<520:
        get_data(i, 6, 1, rf6, 1, pw6, 2, pri6, jitter=jitter6)
    elif 520<= i<540:
        get_data(i, 7, 1, rf7, 1, pw7, 2, pri7, jitter=jitter7)
    elif 540<= i<560:
        get_data(i, 8, 1, rf8, 1, pw8, 2, pri8, jitter=jitter8)


    df = pd.DataFrame(
            data_para, columns=["序号","型号","RF类型","RF1","RF2","RF3","RF4","PW类型",\
                                "PW1","PW2","PW3","PW4","PRI类型","PRI1","PRI2",\
                                "PRI3","PRI4"])
    df.to_csv("E:\\data\\PRI\\1101\\train_20.txt", sep="\t", index=False)  


