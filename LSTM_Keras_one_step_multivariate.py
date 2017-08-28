# -*- coding: utf-8 -*-

from math import sqrt
import matplotlib.pyplot as plt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from utils.get_data_from_mysql import get_data_from_mysql
from utils.get_1_day_max_data import map_str

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.xlabel('Time index for observation points')
    plt.ylabel('P(KW)')
    plt.legend()
    plt.show()
    plt.savefig(filename+'.png')




if __name__=='__main__':
    # load dataset
    epochs  = 50
    month=6
    day_start=20
    day_end=30
    month_start=4
    [day_start_str,day_end_str,month]=map(map_str,[day_start,day_end,month_start])
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    
    start_time='2017-0{0}-{1} 00:00'.format(month_start,day_s~tart_str)
    end_time='2017-0{0}-{1} 23:45'.format(month_start,day_end_str)
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,Ia,Ib,Ic,PFa,PFb,PFc,P'
    dataset=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
    values=dataset.values
    values=values[:,1:]  #把第一列时间索引去掉
    
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7,8,9,10,11,12]], axis=1, inplace=True)
    print(reframed.head())
     
    # split into train and test sets
    values = reframed.values
    n_train_hours = int(values.shape[0]*0.9)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
     
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    inv_y = scaler.inverse_transform(test_X)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    plot_results(inv_yhat,inv_y,'point_by_point_predictions')
    print('Test RMSE: %.3f' % rmse)





