# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
from dateutil.parser import parse
from utils.get_data_from_mysql import get_data_from_mysql

def date_parser(date):
    c=parse(date)
    return c.strftime('%Y-%m-%d')

# PCA分析，提取成分，暂时考虑提取12个主成分
#x_scaler = preprocessing.scale(X)
#pca=PCA(n_components=12,whiten=True,random_state=0)
#xx=pca.fit(x_scaler)
#variances=pca.explained_variance_
#variance_ratio=pca.explained_variance_ratio_
#print '各方向方差：', pca.explained_variance_
#print '方差所占比例：', pca.explained_variance_ratio_

#做ARIMA分析
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
plt.plot(y_obs.values, 'r-', lw=2, label=u'original data')
plt.plot(y_hat.values, 'g-', lw=2, label=u'prediction data')
title = u'the prediction for demand(AR=%d, d=%d, MA=%d):RMSE=%.4f' % (p, d, q, rmse)

plt.legend(loc='upper left')
plt.grid(b=True, ls=':')
plt.title(title, fontsize=11)
plt.tight_layout()
# plt.savefig('%s.png' % title)
plt.show()


