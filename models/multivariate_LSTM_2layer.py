import numpy as np
import matplotlib
from time import time
import matplotlib.pyplot as plt
import pandas as pd 
import math
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Doc du lieu
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('/home/nguyen/LSTM_GoogleTraceData/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

dataset = df.values

# normalize the dataset

length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

RAM = df['mem_usage'].values
CPU = df['cpu_rate'].values
RAM_nomal = scaler.fit_transform(RAM)
CPU_nomal = scaler.fit_transform(CPU)

# create and fit the LSTM network
sliding_widow = [2,3,4,5]
# split into train and test sets
for sliding in sliding_widow:
	print "sliding", sliding
	# Chuan bi du lieu dau vao
	data = []
	for i in range(length-sliding):
		datai=[]
		for j in range(sliding):
			datai.append(CPU_nomal[i+j])
		for j in range(sliding):
			datai.append(RAM_nomal[i+j])
		data.append(datai)
	data = np.array(data)

	train_size = 2880
	test_size = length - train_size
	batch_size_array = [8,16,32,64,128]
	trainX, trainY = data[0:train_size], CPU_nomal[sliding:train_size+sliding]
	testX = data[train_size:length-sliding]
	testY =  CPU[train_size+sliding:length]
	print 'trainX'
	print trainX
	print trainX[0]

	# reshape input to be [samples, time steps, features]

	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	print 'trainX reshape'
	print trainX
	for batch_size in batch_size_array: 
		print "batch_size= ", batch_size
		model = Sequential()
		model.add(LSTM(1, activation = 'relu',input_shape=(1,2*sliding)))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error' ,optimizer='adam' , metrics=['mean_squared_error'])
		history = model.fit(trainX, trainY, epochs=2000, batch_size=batch_size, verbose=2,validation_split=0.1,
		 							callbacks=[EarlyStopping(monitor='loss', patience=20, verbose=1)])
		# make predictions
		# list all data in history
		print(history.history.keys())
		# summarize history for accuracy
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		# plt.show()
		plt.savefig('results/notDecompose/data10minutes/multivariate/cpu/fix_1layer_1neu/history_sliding=%s_batchsize=%s.png'%(sliding,batch_size))
		testPredict = model.predict(testX)

		print len(testPredict), len(testY)
		# invert predictions
		testPredictInverse = scaler.inverse_transform(testPredict)
		print testPredictInverse
		# calculate root mean squared error

		testScoreRMSE = math.sqrt(mean_squared_error(testY, testPredictInverse[:,0]))
		testScoreMAE = mean_absolute_error(testY, testPredictInverse[:,0])
		print('Test Score: %.6f RMSE' % (testScoreRMSE))
		print('Test Score: %.6f MAE' % (testScoreMAE))
		testNotInverseDf = pd.DataFrame(np.array(testPredict))
		testNotInverseDf.to_csv('results/notDecompose/data10minutes/multivariate/cpu/fix_1layer_1neu/testPredict_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		testDf = pd.DataFrame(np.array(testPredictInverse))
		testDf.to_csv('results/notDecompose/data10minutes/multivariate/cpu/fix_1layer_1neu/testPredictInverse_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		errorScore=[]
		errorScore.append(testScoreRMSE)
		errorScore.append(testScoreMAE)
		errorDf = pd.DataFrame(np.array(errorScore))
		errorDf.to_csv('results/notDecompose/data10minutes/multivariate/cpu/fix_1layer_1neu/error_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)