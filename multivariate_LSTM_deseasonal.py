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

# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)

# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph/test.png', histogram_freq=0,  write_graph=True, write_images=True)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

# colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
MemDf = read_csv('data/deseasonalMem10minutes.csv', header=None, index_col=False, names=['mem'], engine='python')
CPUDf = read_csv('data/deseasonalCPU10minutes.csv', header=None, index_col=False, names=['cpu'], engine='python')

# dataset = df.values

# normalize the datase0
length = len(MemDf.values)
scaler = MinMaxScaler(feature_range=(0, 1))

RAM = MemDf['mem'].values
CPU = CPUDf['cpu'].values
RAM_nomal = scaler.fit_transform(RAM)
CPU_nomal = scaler.fit_transform(CPU)




# create and fit the LSTM network
sliding_widow = [2,3,4,5]
# split into train and test sets
for sliding in sliding_widow:
	print "sliding", sliding
	data = []
	for i in range(length-sliding):
		a=[]
		for j in range(sliding):
			a.append(CPU_nomal[i+j])
			a.append(RAM_nomal[i+j])
			# print a
		data.append(a)
	data = np.array(data)
	# split into train and test sets

	# split into train and test sets
	# train_size = int(len(CPU)*0.67)
	train_size = 2880
	test_size = length - train_size
	print train_size
	print test_size
	batch_size_array = [8,16,32,64,128]
	trainX, trainY = data[0:train_size], CPU_nomal[sliding:train_size+sliding]
	testX = data[train_size:length-sliding]
	testY =  CPU[train_size+sliding:length]
	seasonalDf = read_csv('data/seasonalCPU10minutes.csv', names = ['cpu'])
	print "seasonalDf"
	print seasonalDf['cpu'].values
	sesonal = seasonalDf.values[train_size+sliding:length]
	print len(testX)
	print 'trainX'
	print trainX
	print len(testY)
	print len(sesonal)
	# reshape input to be [samples, time steps, features]

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	print 'trainX reshape'
	print trainX
	for batch_size in batch_size_array: 
		print "batch_size= ", batch_size
		model = Sequential()
		model.add(LSTM(32, return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model.add(LSTM(4, activation = 'relu'))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error' ,optimizer='adam' , metrics=['mean_squared_error'])
		history = model.fit(trainX, trainY, epochs=2000, batch_size=batch_size, verbose=2,validation_split=0.1,
		 							callbacks=[EarlyStopping(monitor='loss', patience=20, verbose=1),tensorboard])
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
		plt.savefig('results/Decompose/data10minutes/multivariate/cpu/2layer_32-4neu_timeSteps/history_sliding=%s_batchsize=%s.png'%(sliding,batch_size))
		testPredict = model.predict(testX)

		print 'testPredict', len(testPredict)
		# print testPredict
		print len(sesonal)
		# invert predictions
		testPredictInverse = scaler.inverse_transform(testPredict)
		
		finalPred = []
		for k in range(len(testPredict)):
			finalPred.append(testPredictInverse[k] + sesonal[k])
		print finalPred

		finalPred = np.array(finalPred)
		# calculate root mean squared error

		testScoreRMSE = math.sqrt(mean_squared_error(testY, finalPred))
		testScoreMAE = mean_absolute_error(testY, finalPred)
		print('Test Score: %.6f RMSE' % (testScoreRMSE))
		print('Test Score: %.6f MAE' % (testScoreMAE))
		testNotInverseDf = pd.DataFrame(np.array(testPredict))
		testNotInverseDf.to_csv('results/Decompose/data10minutes/multivariate/cpu/2layer_32-4neu_timeSteps/testPredict_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		testDf = pd.DataFrame(np.array(finalPred))
		testDf.to_csv('results/Decompose/data10minutes/multivariate/cpu/2layer_32-4neu_timeSteps/testPredictInverse_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)
		errorScore=[]
		errorScore.append(testScoreRMSE)
		errorScore.append(testScoreMAE)
		errorDf = pd.DataFrame(np.array(errorScore))
		errorDf.to_csv('results/Decompose/data10minutes/multivariate/cpu/2layer_32-4neu_timeSteps/error_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), index=False, header=None)