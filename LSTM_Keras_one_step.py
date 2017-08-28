# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from utils.get_data_from_mysql import get_data_from_mysql
from utils.get_1_day_max_data import get_1_day_max_data,map_str,merge_month_data


warnings.filterwarnings("ignore")

def load_data(filename_ensem, seq_len, normalise_window):
    tabel_name=filename_ensem[0]
    circuit_name=filename_ensem[1]
    field_name=filename_ensem[2]
    day_start=filename_ensem[3]
    day_end=filename_ensem[4]
    month_start=filename_ensem[5]
    month_end=filename_ensem[6]
    year=filename_ensem[7]
    start_time=filename_ensem[8]
    end_time=filename_ensem[9]
    data_mat_P=get_data_from_mysql(tabel_name,circuit_name,field_name,start_time,end_time)
    data=data_mat_P['P'].values
    
    #提取多个月的每天最大值
#    data=merge_month_data(circuit_name,field_name,year,month_start,month_end,day_start,day_end)
#    data=data['day_max'].values

    scaler=preprocessing.StandardScaler().fit(data)
    data=scaler.transform(data)

    sequence_length = seq_len + 1
    result = [] #为了做分批次训练，result的0维长度就是batch_size
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])  
        #得到长度为seq_len+1的向量，向量中的最后一个作为label,其实相当于用seq_len长度的数据来预测后面1个数据

    result = np.array(result)

    #划分train、test
    row = np.int32(round(0.86 * result.shape[0])) #选出86%作为train
    train = result[:row, :] 
    np.random.shuffle(train) 
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, scaler]


def build_model(layers):  #layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))  #prevent overfitting
    
    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
#    print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    predicted = np.squeeze(predicted)
    return predicted

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
    epochs  = 200
    seq_len = 3  #历史数据的长度
    circuit_name='assetaccount/t_circuit:1479878848817'
    field_name='ACQUISITION_TIME,P'
    year=2017
    month_start=6
    month_end=7
    day_start=2
    day_end=30
    [day_start_str,day_end_str,month]=map(map_str,[day_start,day_end,month_start])
    tabel_name='tb_data_original_2017{0}_new'.format(month)
    
    start_time='2017-0{0}-{1} 00:00'.format(month_start,day_start_str)
    end_time='2017-0{0}-{1} 23:45'.format(month_start,day_end_str)
    filename_ensem=[tabel_name,circuit_name,field_name,day_start,day_end,month_start,month_end,year,start_time,end_time]

    X_train, y_train, X_test, y_test, scaler = load_data(filename_ensem, seq_len, True)

    print('X_train shape:',X_train.shape)  
    print('y_train shape:',y_train.shape) 
    print('X_test shape:',X_test.shape)    
    print('y_test shape:',y_test.shape)    


    model = build_model([1, 50, 100, 1])

    history=model.fit(X_train,y_train,batch_size=512,nb_epoch=epochs,validation_split=0.05)

    point_by_point_predictions = predict_point_by_point(model, X_test)
#    predict_origins=predict_point_by_point(model,X_train)
    print('point_by_point_predictions shape:',np.array(point_by_point_predictions).shape)  #(412L)

    point_by_point_predictions=scaler.inverse_transform(point_by_point_predictions)
    y_test=scaler.inverse_transform(y_test)
    plot_results(point_by_point_predictions,y_test,'point_by_point_predictions')
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, point_by_point_predictions))
    print('Test RMSE: %.3f' % rmse)
    
    
#    plot_results(predict_origins[:50],y_train[:50],'point_by_point_predictions_original')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()