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
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
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
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[1,2], engine='python')

dataset = df.values

# normalize the dataset
length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

CPU_nomal = scaler.fit_transform(dataset.T[0])
RAM_nomal = scaler.fit_transform(dataset.T[1])

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
batch_size_array = [1,2,3,4,5,6,7,8,9,10]
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
	model.add(LSTM(64,activation = 'relu', return_sequences=True,input_shape=(2, 1)))
	model.add(LSTM(32,activation = 'relu', return_sequences=True))
	model.add(LSTM(16))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error' ,optimizer='adam' , metrics=['acc'])
	model.fit(trainX, trainY, epochs=2000, batch_size=batch_size, verbose=2, callbacks=[EarlyStopping(monitor='loss', patience=2, verbose=1),tensorboard])
	# make predictions

	testPredict = model.predict(testX)

	print testPredict
	# invert predictions
	testPredict = scaler.inverse_transform(testPredict)

	# calculate root mean squared error

	testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	testDf = pd.DataFrame(np.array(testPredict))
	testDf.to_csv('results/3layers64-32-16/testPredict_batchsize=%s.csv'%(batch_size), index=False, header=None)
	RMSEScore=[]
	RMSEScore.append(testScore)
	RMSEDf = pd.DataFrame(np.array(RMSEScore))
	RMSEDf.to_csv('results/3layers64-32-16/RMSE_batchsize=%s.csv'%(batch_size), index=False, header=None)