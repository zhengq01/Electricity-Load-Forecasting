# -*- coding: utf-8 -*-

from pandas import Series
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import GridSearchCV
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import time
from utils.get_data_from_mysql import get_data_from_mysql
from utils.get_1_day_max_data import map_str
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = [], []
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out): #cols的list长度就是n_in+n_out
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan: #拿序列中最开始的几个数做样本，他们的t-n值肯定是nan，将这些样本踢掉
		agg.dropna(inplace=True) #丢弃掉的行数是n_lags+n_seq-1，减1是因为提供的序列本身也算作被预测
	return agg
 
# create a differenced series
def difference(dataset, interval=1):
	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_lag, n_seq,len_field):
	# extract raw values
    raw_values=series
	# rescale values to -1, 1
    scaled_values=np.zeros_like(raw_values)
    for i in range(series.shape[1]): #不同的项需要不同的scaler，最后一个scaler用于P，可返回后用于forecast
        scaler = MinMaxScaler() #默认0-1，可以自由设置
        scaled_values[:,i] = scaler.fit_transform(raw_values[:,i])
    
	# transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    columns_to_select=[i for i in range(n_lag*len_field)]+[n_lag*len_field+len_field*i-1 for i in range(1,n_seq+1)]
    supervised_values=supervised_values[:,columns_to_select]
    
	# split into train and test sets
    n_test=np.int32(0.1*round(supervised_values.shape[0]))
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test,n_test
 

def create_model(n_neurons,n_lag,n_seq,len_field):
    model = Sequential()

    model.add(LSTM(n_neurons,batch_input_shape=(None, n_lag, len_field),return_sequences=True))
    model.add(Dropout(0.2))  #prevent overfitting
    
    model.add(LSTM(n_neurons,return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(n_neurons,return_sequences=False))  
    model.add(Dropout(0.3))

    model.add(Dense(n_seq))
    model.add(Activation("linear")) 

    model.compile(loss="mse", optimizer="rmsprop")
    return model
 
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch,n_lag,len_field): 
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, n_lag, len_field)  
	# make forecast
	forecast = model.predict(X, batch_size=n_batch) #batch_size其实比喂进来的样本数大，也不影响
	# convert to list
	return [x for x in forecast[0, :]] 
 
# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_lag, n_seq,len_field):
    forecasts = []
    for i in range(len(test)):
        X=test[i, 0:n_lag*len_field]
		# make forecast
        forecast = forecast_lstm(model, X, n_batch,n_lag,len_field)
#        print forecast
		# store the forecast
        forecasts.append(forecast)
    return forecasts

 
def invboxcox(y,ld):
    if ld == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(ld*y+1)/ld))

# inverse data transform on forecasts
def inverse_transform(forecasts, scaler, n_test):
    inverted = []
    for i in range(len(forecasts)):
        #create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)#将(0,1)的预测值还原到原来的尺度
        inv_scale=list(inv_scale.ravel())
        inverted.append(inv_scale)
    return inverted

  
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):

    plt.rcParams["figure.figsize"] = (11,6)
    plt.plot(series)
	# plot the forecasts in red
    xaxis_mat=[]
    for i in range(n_test):
        off_s = len(series) - len(forecasts[i])-(n_test-1)+i-1
        off_e = off_s + len(forecasts[i])+1
        xaxis = [x for x in range(off_s, off_e)]
        xaxis_mat.append(xaxis)
        yaxis=[series[off_s]]+forecasts[i] #off_s对应的是每个预测序列第一个值的上一个值
        plt.plot(xaxis, yaxis, color='red')
    plt.show()
    
if __name__=='__main__':
    start=time.time()
    # load dataset
    month=4
    day_start=1
    day_end=10
    month_start=4
    [day_start_str,day_end_str,month]=map(map_str,[day_start,day_end,month_start])
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    
    start_time='2017-0{0}-{1} 00:00'.format(month_start,day_start_str)
    end_time='2017-0{0}-{1} 23:45'.format(month_start,day_end_str)
    
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,Ia,Ib,Ic,PFa,PFb,PFc,P'
    len_field=len(field_name.split(','))-1

    data_mat_P=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
    data_mat_P=data_mat_P.values
    series=data_mat_P[:,1:].astype('float32')
       
    # configure
    n_lag = 8
    n_seq = 5
    epochs = 150
    n_batch = 50 
    n_neurons=50
    # prepare data
    scaler, train, test,n_test = prepare_data(series, n_lag, n_seq,len_field)
       
    #GridSearch for optimal hyperparameters
#    X_train, y_train = train[:, :n_lag], train[:, n_lag:]
#    model=KerasClassifier(build_fn=create_model,verbose=0)
#    batch_size=[10,20,40,50,60,80,100]
#    epochs=[20,50,100,150]
#    neurons=[10,20,30,50]
#    param_grid=dict(n_neurons=neurons)
#    grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1)
#    X_train=X_train.reshape((-1,n_lag,1))
#    grid_result=grid.fit(X_train,y_train)
#    
#    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#    means = grid_result.cv_results_['mean_test_score']
#    stds = grid_result.cv_results_['std_test_score']
#    params = grid_result.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))
        
    # fit model
    layers=[n_lag,50,100,n_seq]
    model=create_model(n_neurons,n_lag,n_seq,len_field)
    X_train, y_train = train[:, :n_lag*len_field], train[:, n_lag*len_field:]
    X_train = X_train.reshape(X_train.shape[0], n_lag, len_field)
    history=model.fit(X_train,y_train,batch_size=n_batch,shuffle=False,epochs=epochs,validation_split=0.05,verbose=2)
    
    # make forecasts
    forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq,len_field)
#    forecasts_origin=make_forecasts(model, n_batch, train, n_lag, n_seq)
    
    # inverse transform forecasts and test
    forecasts = inverse_transform(forecasts, scaler, n_test)
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(actual, scaler, n_test)
    
#    forecasts1 = inverse_transform(series, forecasts_origin, scaler, n_test+train.shape[0]+2)
#    actual1 = [row[n_lag:] for row in train]
#    actual1 = inverse_transform(series, actual1, scaler, n_test+train.shape[0]+2)

    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, n_lag, n_seq)
    
    # plot forecasts
    plot_forecasts(series[-200:,-1], forecasts, n_test)
#    plot_forecasts(series, forecasts1, n_test+train.shape[0]+2)
    
    time_elapse=time.time()-start
    print time_elapse
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    

