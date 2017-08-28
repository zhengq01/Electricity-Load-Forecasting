# -*- coding: utf-8 -*-

import numpy as np


def generate_hour_data(data):
    data_ensem=[]
    for i in range(len(data)/4):  #每个小时4个数，所以每隔4个数取一个均值
        data_ensem.append(np.mean(data[i*4:(i+1)*4]))
    data_ensem=np.array(data_ensem)
    return data_ensem