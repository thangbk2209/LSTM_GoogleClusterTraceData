import numpy as np
import matplotlib
from time import time
matplotlib.use('Agg')
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

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

dataset = df.values

# normalize the dataset
length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

RAM_nomal = scaler.fit_transform(dataset.T[1])
CPU_nomal = scaler.fit_transform(dataset.T[0])




sliding_widow = [1,2,3,4,5]
# split into train and test sets
for sliding in sliding_widow:
	print "sliding", sliding
	train_size = int(length * 0.67)
	test_size = length - train_size
	batch_size_array = [8,16,32,64,128]
	data = []
	for i in range(length-sliding):
		a=[]
		for j in range(sliding):
			a.append(CPU_nomal[i+j])
			a.append(RAM_nomal[i+j])
			# print a
		data.append(a)
	data = np.array(data)
	print data [0]
	# print data
	trainX  = data[0:train_size]
	print trainX
	trainY = CPU_nomal[sliding:train_size+sliding]
	testX = data[train_size:length-sliding]
	testY =  dataset.T[1][train_size+sliding:length]
	# reshape input to be [samples, time steps, features]
	print "testx,testy"
	print testX[0],testY[0]
	print testX[1],testY[1]
	trainX = np.reshape(trainX, (-1, trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	print trainX
	# create and fit the LSTM network
	for batch_size in batch_size_array: 
		print "batch_size= ", batch_size
		model = Sequential()
		model.add(LSTM(32,return_sequences=True,input_shape=(2*sliding, 1)))
		model.add(LSTM(4))
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
		plt.savefig('resultsdata5minutes/2layer_32-4neuTimesteps/history_sliding=%s_batchsize=%s.png'%(sliding,batch_size))
		testPredict = model.predict(testX)

		print testPredict
		# invert predictions
		testPredictInverse = scaler.inverse_transform(testPredict)
		print testPredictInverse
		# calculate root mean squared error

		testScoreRMSE = math.sqrt(mean_squared_error(testY, testPredictInverse[:,0]))
		testScoreMAE = mean_absolute_error(testY, testPredictInverse[:,0])
		print('Test Score: %.2f RMSE' % (testScoreRMSE))
		print('Test Score: %.2f MAE' % (testScoreMAE))
		testNotInverseDf = pd.DataFrame(np.array(testPredict))
		testNotInverseDf.to_csv('resultsdata5minutes/2layer_32-4neuTimesteps/testPredict_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		testDf = pd.DataFrame(np.array(testPredictInverse))
		testDf.to_csv('resultsdata5minutes/2layer_32-4neuTimesteps/testPredictInverse_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		errorScore=[]
		errorScore.append(testScoreRMSE)
		errorScore.append(testScoreMAE)
		errorDf = pd.DataFrame(np.array(errorScore))
		errorDf.to_csv('resultsdata5minutes/2layer_32-4neuTimesteps/error_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)