# Electricity Load Forecasting
* ARIMA.py是用ARIMA做的时间序列分析
* LSTM_Keras_one_step.py是用LSTM做的单步时间序列分析
* LSTM_Keras_one_step_multivariate.py在单步预测的基础上，考虑更多的因素，比如三相电流、三相有功功率因数等等
* LSTM_Keras_multi_steps_multivariate.py在单步多因素预测的基础上，考虑多步预测
* GBDT_forecast.py是用GBDT实现了考虑了多因素的负荷预测，Xgboost_forecast.py与之类似
* taizhou_weather.csv和hangzhou_weather.csv是台州和杭州1-7月份的每日最低和最高气温数据
* utils是一些被调函数
