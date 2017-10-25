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
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, engine='python')

dataset = df.values

# normalize the dataset
length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

meanCPU = np.mean(df['cpu_rate'].values)
maxCPU = np.max(df['cpu_rate'].values)
minCPU = np.min(df['cpu_rate'].values)
print meanCPU
print maxCPU

CPU_nomal=[]
for i in range(len(df['cpu_rate'].values)):
	cpu = (df['cpu_rate'].values[i] - minCPU)/(maxCPU - minCPU)
	CPU_nomal.append(cpu)
# CPU_nomal = scaler.fit_transform(dataset.T[0])
# RAM_nomal = scaler.fit_transform(dataset.T[1])
print CPU_nomal

meanMem = np.mean(df['mem_usage'].values)
maxMem = np.max(df['mem_usage'].values)
minMem = np.min(df['mem_usage'].values)
RAM_nomal=[]
for i in range(len(df['mem_usage'].values)):
	ram = (df['mem_usage'].values[i] - minMem)/(maxMem - minMem)
	RAM_nomal.append(ram)
# CPU_nomal = scaler.fit_transform(dataset.T[0])
# RAM_nomal = scaler.fit_transform(dataset.T[1])
# print CPU_nomal
data = []
for i in range(length):
	a=[]
	a.append(CPU_nomal[i])
	a.append(RAM_nomal[i])
	data.append(a)
data = np.array(data)

# split into train and test sets

# split into train and test sets
train_size = int(length * 0.67)
test_size = length - train_size
batch_size_array = [8,16,32,64,128]
trainX, trainY = data[0:train_size], CPU_nomal[0:train_size]
testX = data[train_size:length]
testY =  dataset.T[1][train_size:length]
# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
for batch_size in batch_size_array: 
	print "batch_size= ", batch_size
	model = Sequential()
	model.add(LSTM(4,return_sequences=True, activation = 'relu',input_shape=(2, 1)))
	model.add(LSTM(2))
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
	plt.savefig('results/2layer_4-2neu/history_batchsize=%s.png'%(batch_size))
	testPredict = model.predict(testX)

	print testPredict
	# invert predictions
	testPredictInverse = []
	for pred in testPredict:
		testPredictInverse.append(pred * (maxCPU-minCPU) + minCPU)
	print testPredictInverse
	# calculate root mean squared error

	testScoreRMSE = math.sqrt(mean_squared_error(testY, testPredictInverse[:,0]))
	testScoreMAE = mean_absolute_error(testY, testPredictInverse[:,0])
	print('Test Score: %.2f RMSE' % (testScoreRMSE))
	print('Test Score: %.2f MAE' % (testScoreMAE))
	testDf = pd.DataFrame(np.array(testPredict))
	testDf.to_csv('results/2layer_4-2neu/testPredict_batchsize=%s.csv'%(batch_size), index=False, header=None)
	testInverseDf = pd.DataFrame(np.array(testPredictInverse))
	testInverseDf.to_csv('results/2layer_4-2neu/testPredictInverse_batchsize=%s.csv'%(batch_size), index=False, header=None)
	errorScore=[]
	errorScore.append(testScoreRMSE)
	errorScore.append(testScoreMAE)
	errorDf = pd.DataFrame(np.array(errorScore))
	errorDf.to_csv('results/2layer_4-2neu/error_batchsize=%s.csv'%(batch_size), index=False, header=None)