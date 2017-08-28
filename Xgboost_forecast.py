# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from utils.get_1_day_max_data import get_1_day_max_data,merge_month_data

#获取天气数据
weather_data=pd.read_csv('taizhou_weather.csv',skiprows=1,usecols=[0,2,3],names=['date','T_min','T_max'])
weather_data['date']=pd.to_datetime(weather_data['date'])
weather_data['date']=weather_data['date'].apply(lambda x:x.strftime('%Y-%m-%d'))

#获取逐日的最大负荷
circuit_name='assetaccount/t_circuit:1479711216494'
field_name='ACQUISITION_TIME,P'
day_start=1
day_end=30
month_start=1
month_end=7
year=2017
day_max_data=merge_month_data(circuit_name,field_name,year,month_start,month_end,day_start,day_end)
day_max_data=day_max_data.set_index(np.arange(day_max_data.shape[0]))

#保证负荷数据和天气数据按时间对应，merge的过程取并集，
total_data=pd.merge(day_max_data,weather_data,left_on='day_index',right_on='date',how='outer')
del total_data['date']
total_data['day_index']=pd.to_datetime(total_data['day_index'])
nan_index_p=list(total_data[total_data.loc[:,'day_max'].isnull()==True].index.values)
nan_index_T=list(total_data[total_data.loc[:,'T_max'].isnull()==True].index.values)
nan_index_p.extend(nan_index_T)
for i in nan_index_p:
    for column in ['day_max','T_min','T_max']: #这里用nan的前后两个值平均去填充，但如果有两个连着的nan就不行了
        total_data.loc[i,column]=(total_data.loc[i-1,column]+total_data.loc[i-1,column])/2
    
# 补充特征
total_data['dow'] = total_data['day_index'].apply(lambda x: x.dayofweek)
total_data['doy'] = total_data['day_index'].apply(lambda x: x.dayofyear)
total_data['day'] = total_data['day_index'].apply(lambda x: x.day)
total_data['month'] = total_data['day_index'].apply(lambda x: x.month)

def map_season(month):
    month_dic = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:3, 9:3, 10:4, 11:4, 12:1}
    return month_dic[month]

total_data['season'] =total_data['month'].apply(lambda x: map_season(x))

total_data['day_1']=total_data.loc[:,'day_max'].shift(1)
total_data['day_2']=total_data.loc[:,'day_max'].shift(2)
total_data['day_3']=total_data.loc[:,'day_max'].shift(3)

total_data_2=total_data[~pd.isnull(total_data['day_3'])]

##去除春节时间段的数据，春节阶段用电是不常规的，并且预测阶段没有春节
a_=total_data_2[total_data_2.day_index>'2017-01-18'].index.values
b_=total_data_2[total_data_2.day_index<'2017-02-08'].index.values
spring_festival_index=list(set(a_).intersection(set(b_)))
total_data_2=total_data_2.drop(spring_festival_index)

#构造训练集和测试集
train_data=total_data_2[total_data_2.day_index<'2017-07-01']
test_data=total_data_2[total_data_2.day_index>='2017-07-01']


train_lgb=train_data.copy()
train_lgb[['dow','doy','day','month','season']]=train_lgb[['dow','doy','day','month','season']].astype('str')
test_lgb=test_data.copy()
test_lgb[['dow','doy','day','month','season']]=test_lgb[['dow','doy','day','month','season']].astype('str')

X_train=train_lgb[['T_min','T_max','dow','doy','day','month','season','day_1','day_2','day_3']]
y_train=np.squeeze(train_lgb[['day_max']].values)
X_test=test_lgb[['T_min','T_max','dow','doy','day','month','season','day_1','day_2','day_3']]
y_test=np.squeeze(test_lgb[['day_max']].values)



##模型超参数选择
#model=XGBRegressor()
#param_grid = {
#    'learning_rate': [0.01, 0.02, 0.05, 0.1,0.2],
#    'n_estimators': [100, 200, 400, 800, 1000],
#    'max_depth':[3,4,5,6]}
##fit_params = {'categorical_feature':[2,3,4,5,6]}
#
##kfold=StratifiedKFold(n_splits=4,shuffle=True,random_state=7)
#grid_search = GridSearchCV(model, param_grid,n_jobs=-1,verbose=1)
#
#grid_result=grid_search.fit(X_train, y_train)
#print('Best parameters found by grid search are:', grid_result.best_params_)
#print grid_result.best_score_


##模型训练
X_train.columns=np.arange(X_train.shape[1])
X_test.columns=np.arange(X_test.shape[1])
X_train,X_test=map(lambda a:np.array(a),[X_train,X_test])
data_train = xgb.DMatrix(X_train, label=y_train)
data_test = xgb.DMatrix(X_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth':6, 'eta': 0.1, 'silent': 1, 'objective': 'reg:linear'}

bst = xgb.train(param, data_train, num_boost_round=60, evals=watch_list)

#绘制特征的重要性分布
xgb.plot_importance(bst)


##训练部分的拟合效果展示
plt.figure()
y_train_hat=bst.predict(data_train)
plt.plot(y_train_hat,'b',label='Prediction')
plt.plot(y_train,'r',label='True data')
plt.xlabel('Time index for observation points')
plt.ylabel('P(KW)')
print 'rmse for training:',np.sqrt(mean_squared_error(y_train_hat,y_train))
plt.legend()
plt.show()

##预测
plt.figure()
y_predict=bst.predict(data_test)
print 'rmse for testing:',np.sqrt(mean_squared_error(y_predict,y_test))
plt.plot(y_predict,'b',label='Prediction')
plt.plot(y_test,'r',label='True data')
plt.xlabel('Time index for observation points')
plt.ylabel('P(KW)')
plt.legend()
plt.show()


plt.figure()
[y_train_hat,y_train,y_predict,y_test]=map(lambda x:list(x),[y_train_hat,y_train,y_predict,y_test])
y_train_hat.extend(y_predict)
y_train.extend(y_test)
plt.plot(y_train_hat[-50:],'b',label='Prediction')
plt.plot(y_train[-50:],'r',label='True data')
plt.xlabel('Time index for observation points')
plt.ylabel('P(KW)')
plt.legend()
plt.show()