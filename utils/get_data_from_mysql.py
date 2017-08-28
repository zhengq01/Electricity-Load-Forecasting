# -*- coding: utf-8 -*-

import MySQLdb
import numpy as np
import pandas as pd

def get_data_from_mysql(tabel_name,circuit_name,field_name,start,end):
    
    db=MySQLdb.connect(host='hzzh-data.mysql.rds.aliyuncs.com', user='reader', passwd='2017$Mysql', db='lego', port=3306, charset='utf8')
    cursor=db.cursor()
    
#    query="""
#    SELECT
#    	ACQUISITION_TIME,P    #需要获取的字段数据，在这边修改即可
#    FROM
#    	tb_data_original_201706_new a
#    WHERE
#    	a.MONITOR_POINT_GUID = 'assetaccount/t_circuit:1479471003360‘
#    """
#    tabel_name='tb_data_original_201706_new'
#    circuit_name='assetaccount/t_circuit:1479471003360'
#    field_name='ACQUISITION_TIME,P'
    
    query=""" SELECT {0}    
    FROM
    	{1} a
    WHERE
    	a.MONITOR_POINT_GUID = '{2}'
    and a.ACQUISITION_TIME BETWEEN '{3}' AND '{4}'
    """.format(field_name,tabel_name,circuit_name,start,end)

    cursor.execute(query)
    data=cursor.fetchall()
    db.close()
    industry_stat={}
    for index,name in enumerate(field_name.split(',')):
        temp=[]
        for i in xrange(len(data)):
            if data[i][1] is None:
                continue
            if index==0:
                temp.append(data[i][index].encode('utf-8'))
            else:
                temp.append(np.float64(data[i][index]))
        industry_stat[name]=temp
                
    data_mat_P=pd.DataFrame(industry_stat,columns=field_name.split(',')) 
    return data_mat_P

def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))


if __name__=='__main__':
    tabel_name='tb_data_original_201706_new'
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,P'
    start='2017-01-02 00:00'
    end='2017-01-21 23:45'
    aa1=get_data_from_mysql(tabel_name,circuit_name,field_name,start,end)
    bb=aa1['P'].values

    
    