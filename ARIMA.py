# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
from dateutil.parser import parse
from utils.get_data_from_mysql import get_data_from_mysql

def date_parser(date):
    c=parse(date)
    return c.strftime('%Y-%m-%d')

#导入数据，处理掉缺失值
#header_for_original=['GUID_','Monitor_point_GUID','Acquisition_time','Ua','Ub','Uc','Ia','Ib',
# 'Ic','PF','PFa','PFb','PFc','P','Pa','Pb','Pc','Q','Qa','Qb','Qc','Direction_EP_sum','Direction EP','Direction EP original','Direction_EQ_sum','Direction EQ','Reverse_EP','Reverse_EQ_sum','Reverse_EQ','F','D']
#header_for_demand=['GUID','Monitor_point_GUID','Acquisition_time','D']
##data_original_=pd.read_csv('tb_data_original_201706_new_1478603483832.csv',names=header_for_original,parse_dates=['Acquisition_time'], date_parser=date_parser)
#data_demand_mat=pd.DataFrame(np.arange(2).reshape(1,-1),columns=header_for_demand[2:])
#for i in range(2,7):
#    if i==5: #第5个月的数据有重复
#        temp=pd.read_csv('../data/tb_data_original_demand_20170{0}_new_1479274058119.csv'.format(i),names=header_for_demand,parse_dates=['Acquisition_time'], date_parser=date_parser)
#        temp=temp.groupby('Acquisition_time')
#        data_demand_=temp.mean()
#        data_demand_mat.append(data_demand_)
#    else:
#        data_demand_=pd.read_csv('../data/tb_data_original_demand_20170{0}_new_1479274058119.csv'.format(i),names=header_for_demand,parse_dates=['Acquisition_time'], date_parser=date_parser)
#        data_demand_=data_demand_.iloc[:,2:]
#        data_demand_mat=data_demand_mat.append(data_demand_)
#data_demand_mat=data_demand_mat[1:]
#Y_1=data_demand_mat.sort_values(['Acquisition_time'])
#Y=Y_1[9:] #不知为何放了第个时间点前面的数据的时候就没有预测值了
#data_demand=data_demand_[62:93]
#data_original_=data_original_[1:]
#colume_for_del=[] #待删除的列
#for colume_name in header_for_original:
#    if data_original_[colume_name].count()<0.9*data_original_.shape[0]: #认为如果非缺失值的个数小于90%，那么丢弃这个特征
#        colume_for_del.append(colume_name)
#data_original_2=data_original_.drop(colume_for_del,axis=1)

#print data_original_2[data_original_2.isnull().values==True]


# PCA分析，提取成分，暂时考虑提取12个主成分
#x_scaler = preprocessing.scale(X)
#pca=PCA(n_components=12,whiten=True,random_state=0)
#xx=pca.fit(x_scaler)
#variances=pca.explained_variance_
#variance_ratio=pca.explained_variance_ratio_
#print '各方向方差：', pca.explained_variance_
#print '方差所占比例：', pca.explained_variance_ratio_

# 做ARIMA分析
epochs  = 80
seq_len = 4
tabel_name='tb_data_original_201706_new'
circuit_name='assetaccount/t_circuit:1479878848817'
field_name='ACQUISITION_TIME,P'
start_time='2017-06-01 00:00'
end_time='2017-06-10 23:45'
Y=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
y=Y.set_index('Acquisition_time'.upper())
y_obs=y['P'].astype(np.float)
y_obs=np.log(y_obs)

show = 'prime'   # 'diff', 'ma', 'prime'
d = 1
diff = y_obs - y_obs.shift(periods=d)
ma = y_obs.rolling(window=3).mean()
y_obs_ma = y_obs - ma
p = 2
q = 2
model = ARIMA(endog=y_obs, order=(p, d, q))     # 自回归函数p,差分d,移动平均数q
arima = model.fit(disp=-1)                  # disp<0:不输出过程
prediction = arima.fittedvalues
print type(prediction)
y_hat = prediction.cumsum() + y_obs[0]
mse = ((y_obs - y_hat)**2).mean()
rmse = np.sqrt(mse)


plt.figure(facecolor='w')
if show == 'diff':
    plt.plot(y_obs, 'r-', lw=2, label=u'original data')
    plt.plot(diff, 'g-', lw=2, label=u'%d order diff' % d)
    #plt.plot(prediction, 'r-', lw=2, label=u'预测数据')
    title = u'the curve of demand - Log transform'
elif show == 'ma':
    #plt.plot(x, 'r-', lw=2, label=u'原始数据')
    #plt.plot(ma, 'g-', lw=2, label=u'滑动平均数据')
    plt.plot(y_obs_ma, 'g-', lw=2, label=u'ln(original data) - ln(moving average)')
    plt.plot(prediction, 'r-', lw=2, label=u'prediction data')
    title = u'moving average and MA prediction'
else:
    plt.plot(y_obs.values, 'r-', lw=2, label=u'original data')
    plt.plot(y_hat.values, 'g-', lw=2, label=u'prediction data')
    title = u'the prediction for demand(AR=%d, d=%d, MA=%d):RMSE=%.4f' % (p, d, q, rmse)
plt.legend(loc='upper left')
plt.grid(b=True, ls=':')
plt.title(title, fontsize=11)
plt.tight_layout()
# plt.savefig('%s.png' % title)
plt.show()




