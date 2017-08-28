# -*- coding: utf-8 -*-

#对序列直接编码，参考范围是1-100
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



def one_hot_encode(X, series_min,series_max,n_unique=100):

    gap=(series_max-series_min)/n_unique
    Xenc=[]
    for sequence in X:
        new_index_ensem=[]
        for value in sequence:
            new_index=(value-series_min)/gap
            new_index_ensem.append(int(new_index))
        encoding=[]
        for index in new_index_ensem:
            vector=[0 for _ in range(n_unique)]
            vector[index]=1
            encoding.append(vector)
        Xenc.append(encoding)
    return np.array(Xenc)

# decode a one hot encoded string
def one_hot_decode(y,series_min,series_max,n_unique=100):
    gap=(series_max-series_min)/n_unique
    y_dec=[]
    for encoded_seq in y:
        decoded_seq=[argmax(vector) for vector in encoded_seq]
        decoded_seq=np.array(decoded_seq)
        decoded_seq_tran=list(decoded_seq*gap+series_min)
        y_dec.append(decoded_seq_tran)
    return y_dec


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
    series_min=min(series)
    series_max=max(series)
    series=series.reshape(-1,1)
    n_in=8
    n_out=5
    n_numbers = n_in
    largest = 10
    encoded_length=100
    
    supervised_data=series_to_supervised(series,n_in,n_out)
    supervised_data=supervised_data.values
    n_test=np.int32(0.1*round(supervised_data.shape[0]))
    train, test = supervised_data[0:-n_test], supervised_data[-n_test:]
    X_train, y_train = train[:, :n_in], train[:, n_in:]
    [X_train,y_train]=map(lambda a:list(a),[X_train,y_train])
    X_train=map(lambda a:list(a),X_train)
    y_train=map(lambda a:list(a),y_train)
    X=one_hot_encode(X_train,series_min,series_max)
    y=one_hot_encode(y_train,series_min,series_max)
    
    
    # define LSTM configuration
    n_batch = 50
    n_epoch = 300
    batch_size=50
    n_in_seq_length=len(list(X[0]))
    n_out_seq_length=len(list(y[0]))
   
    # create LSTM
    model = Sequential()
    model.add(LSTM(150, batch_input_shape=(None,n_in_seq_length, encoded_length)))  #encoder
    model.add(Dropout(0.2))
    model.add(RepeatVector(n_out_seq_length))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
   
    # train LSTM
    history=model.fit(X, y, epochs=n_epoch, batch_size=n_batch,validation_split=0.05,shuffle=False,verbose=2)

 
    # evaluate on some new patterns
    X_test, y_test= test[:, :n_in], test[:, n_in:]
    [X_test,y_test]=map(lambda a:list(a),[X_test,y_test])
    X_test=map(lambda a:list(a),X_test)
    y_test=map(lambda a:list(a),y_test)
    X_test=one_hot_encode(X_test,series_min,series_max)
    y_test=one_hot_encode(y_test,series_min,series_max)

    result = model.predict(X_test, batch_size=n_batch, verbose=0)
  
    # calculate error
    expected = one_hot_decode(y_test,series_min,series_max)
    predicted = one_hot_decode(result,series_min,series_max)
    
    
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
    

