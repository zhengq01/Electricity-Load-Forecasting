# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from get_data_from_mysql import get_data_from_mysql
from dateutil.parser import parse


def date_parser(date):
    c=parse(date)
    return c.strftime('%d')

#data_original=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
#time_=data_original['ACQUISITION_TIME']
#data_original['day_index']=time_.apply(date_parser)
#
#day_list=[]
#for i in xrange(1,31):
#    if i<10:
#        str_='0'+str(i)
#    else:
#        str_=str(i)
#    day_list.append(str_)
#
#day_max_list=[]
#for day in day_list:
#    day_max_1=data_original[data_original.loc[:,'day_index']==day].max()
#    day_max_list.append(day_max_1['P'])
#
#day_max=pd.Series(day_max_list,name='day_max')
#day_max=day_max.fillna(day_max.mean())

def map_str(number):
    if number<10:
        number_str='0'+str(number)
    else:
        number_str=str(number)
    return number_str


def get_1_day_max_data(circuit_name,field_name,year,month,day_start,day_end):
    """
    提取每个月的每天的最大值，返回2列的dataframe
    """
    month=map_str(month)
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    [month_str,day_start_str,day_end_str]=map(map_str,[month,day_start,day_end])
    
    start_time='{0}-{1}-{2} 00:00'.format(year,month_str,day_start_str)
    end_time='{0}-{1}-{2} 23:45'.format(year,month_str,day_end_str)

    data_original=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
    time_=data_original['ACQUISITION_TIME']
    data_original['day_index']=time_.apply(date_parser)
#    return data_original
    day_list=[]
    for i in xrange(day_start,day_end+1):
        if i<10:
            str_='0'+str(i)
        else:
            str_=str(i)
        day_list.append(str_)
    
    day_max_list=[]
    for day in day_list:
        day_max_1=data_original[data_original.loc[:,'day_index']==day].max()
        day_max_list.append(day_max_1['P'])
    
    day_max=pd.DataFrame({'day_index':np.arange(1,len(day_max_list)+1),'day_max':day_max_list})
    start='{0}-{1}-{2}'.format(year,month,day_start)
    end='{0}-{1}-{2}'.format(year,month,day_end)
    day_max['day_index']=pd.date_range(pd.to_datetime(start),pd.to_datetime(end))
    day_max['day_index']=day_max['day_index'].apply(lambda x:x.strftime('%Y-%m-%d'))
    day_max['day_max']=day_max['day_max'].fillna(day_max['day_max'].mean())
    return day_max

def merge_month_data(circuit_name,field_name,year,month_start,month_end,day_start,day_end):
    """
    拼接不同月份的每天最大值
    """
    day_max_data=pd.DataFrame(np.arange(2).reshape((1,-1)),columns=['day_index','day_max'])
    for month in range(month_start,month_end+1):
        if month==2:
            day_end=28
        elif month in [1,3,5,7,8,10,12]:
            day_end=31
        else:
            day_end=30
        temp=get_1_day_max_data(circuit_name,field_name,year,month,day_start,day_end)
        temp=temp.sort_values('day_index')
        day_max_data=day_max_data.append(temp)
    day_max_data=day_max_data[1:]
    day_max_data=day_max_data.set_index(np.arange(day_max_data.shape[0]))
    return day_max_data

if __name__=='__main__':
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,P'
    year=2017
    month_start=1
    month_end=7
    
    day_start=1
    day_end=30  
#    day_max=get_1_day_max_data(circuit_name,field_name,year,month_start,day_start,day_end)
    data=merge_month_data(circuit_name,field_name,year,month_start,month_end,day_start,day_end)
#



