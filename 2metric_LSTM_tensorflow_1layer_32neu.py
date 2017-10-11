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
colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# 
df = read_csv('/home/hunter/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4,6], engine='python')
# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
# df = read_csv('/home/nguyen/learnRNNs/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4,6], engine='python')

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
train, test = data[0:train_size,:], data[train_size:length,:]
print "train"
print train
print test
# reshape into X=t and Y=t+1
# look_back = 3
trainX, trainY = data[0:train_size], CPU_nomal[0:train_size]
testX, testY = data[train_size:length], CPU_nomal[train_size:length]
# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4,input_shape=(2, 1)))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam' , metrics=['acc'])
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2, callbacks=[tensorboard])
# make predictions


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print trainPredict
print testPredict
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainDf = pd.DataFrame(np.array(trainPredict))
trainDf.to_csv('results/2metric_LSTM_tensorflow_1layer_4neu_trainPredict.csv', index=False, header=None)

testDf = pd.DataFrame(np.array(testPredict))
testDf.to_csv('results/2metric_LSTM_tensorflow_1layer_4neu_testPredict.csv', index=False, header=None)
RMSEScore=[]
RMSEScore.append(trainScore)
RMSEScore.append(testScore)
RMSEDf = pd.DataFrame(np.array(RMSEScore))
RMSEDf.to_csv('results/2metric_LSTM_tensorflow_1layer_4neu.csv', index=False, header=None)