# -*- coding: utf-8 -*-

import MySQLdb
import numpy as np
import pandas as pd
from get_1_day_max_data import get_1_day_max_data

def get_3_day_max_data(circuit_name,field_name,month_start,month_end):
    year=2017
    day_max_mat=pd.DataFrame(np.arange(2).reshape(1,2),columns=['day_index','day_max'])
    for month in range(month_start,month_end+1):
        day_start=1
        if month==2:
            day_end=28
        else:
            day_end=30
        tabel_name='tb_data_original_20170{0}_new'.format(month)
        day_max_=get_1_day_max_data(tabel_name,circuit_name,field_name,year,month,day_start,day_end)
        day_max_mat=pd.concat([day_max_mat,day_max_])
    day_max_mat=day_max_mat[1:]
    day_max_mat=day_max_mat.set_index(np.arange(day_max_mat.shape[0]))
    day_max_mat=day_max_mat['day_max']
    day_1_max=day_max_mat.values
#    day_3_max=[]
#    for i in range(day_max_mat.shape[0]/3):
#        temp=max(day_max_mat[i*3:(i+1)*3])
#        day_3_max.append(temp)
    return day_1_max
#    return day_3_max
    
if __name__=='__main__':
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,P'
    month_start=1
    month_end=7
    day_3_max_mat=get_3_day_max_data(circuit_name,field_name,month_start,month_end)