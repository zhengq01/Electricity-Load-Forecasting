# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pandas import DataFrame
from pandas import concat
from numpy import argmax
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error
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
    
    agg=agg.applymap(lambda x:np.int32(x))
    return agg

 
# convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = 3
    Xstr = []
    for pattern in X:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        Xstr.append(element_ensem) 
    
    ystr=[]
    for pattern in y:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        ystr.append(element_ensem)

    return Xstr,ystr

# integer encode strings
def integer_encode(X, y, alphabet):
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = []
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]  #返回的是每个字符在alphabet的索引
		Xenc.append(integer_encoded)
	yenc = []
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
		yenc.append(integer_encoded)
	return Xenc, yenc
 
# one hot encode
def one_hot_encode(X, y, max_int):
	Xenc = []
	for seq in X:
		pattern = []
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		Xenc.append(pattern)
	yenc = []
	for seq in y:
		pattern = []
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		yenc.append(pattern)
	return Xenc, yenc
 

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
        
        
# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = []
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    strings_ensem=''.join(strings)
    strings_split=strings_ensem.split(',')
    bb=[aa.strip() for aa in strings_split]
    bb=[int(aa) for aa in bb]
    return bb

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
    # load dataset
    day_start=1
    day_end=10
    month_start=4
    [day_start_str,day_end_str,month]=map(map_str,[day_start,day_end,month_start])
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    field_name='ACQUISITION_TIME,P'
    start_time='2017-0{0}-{1} 00:00'.format(month_start,day_start_str)
    end_time='2017-0{0}-{1} 23:45'.format(month_start,day_end_str)
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    circuit_name='assetaccount/t_circuit:1479878848817'
    
    data_mat_P=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
    series=data_mat_P['P'].values
    series=series.reshape(-1,1)
    n_in=8
    n_out=5
    n_numbers = n_in
    largest = 10
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',',' ']
    n_chars = len(alphabet)
    
    supervised_data=series_to_supervised(series,n_in,n_out)
    supervised_data=supervised_data.values
    n_test=np.int32(0.1*round(supervised_data.shape[0]))
    train, test = supervised_data[0:-n_test], supervised_data[-n_test:]
    X_train, y_train = train[:, :n_in], train[:, n_in:]
    [X_train,y_train]=map(lambda a:list(a),[X_train,y_train])
    X_train=map(lambda a:list(a),X_train)
    y_train=map(lambda a:list(a),y_train)
    X_str,y_str=to_string(X_train,y_train,n_numbers,largest)
    X, y = integer_encode(X_str, y_str, alphabet)
    X, y = one_hot_encode(X, y, len(alphabet))
    
    
    # define LSTM configuration
    n_batch = 50
    n_epoch = 100
    n_in_seq_length=len(list(X[0]))
    n_out_seq_length=len(list(y[0]))
   
    # create LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))  #encoder
    model.add(Dropout(0.2))
    model.add(RepeatVector(n_out_seq_length))
    model.add(LSTM(50, return_sequences=True))  #decoder
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # train LSTM
    history=model.fit(X, y, epochs=n_epoch, batch_size=n_batch,validation_split=0.05,shuffle=False)

    # evaluate on some new patterns
    X_test, y_test= test[:, :n_in], test[:, n_in:]
    [X_test,y_test]=map(lambda a:list(a),[X_test,y_test])
    X_test=map(lambda a:list(a),X_test)
    y_test=map(lambda a:list(a),y_test)
    X_test_str,y_test_str=to_string(X_test,y_test,8,10)
    X_test, y_test = integer_encode(X_test_str, y_test_str, alphabet)
    X_test, y_test = one_hot_encode(X_test, y_test, len(alphabet))

    result = model.predict(X_test, batch_size=n_batch, verbose=0)
  
    # calculate error
    expected = [invert(x, alphabet) for x in y_test]
    predicted = [invert(x, alphabet) for x in result]
    
    
    evaluate_forecasts(expected,predicted,n_in,n_out)
    series=series.ravel()
    plot_forecasts(series[-120:], predicted, n_test)
#    [expected,predicted]=map(lambda x:np.array(x),[expected,predicted])
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    

