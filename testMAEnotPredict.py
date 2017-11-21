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
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

dataset = df.values

# normalize the datase0
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
	train_size = 2880
	test_size = length - train_size
	trainX, trainY = data[0:train_size], CPU_nomal[sliding:train_size+sliding]
	testX = data[train_size:length-sliding]
	testY =  CPU[train_size+sliding:length]
	# pred = CPU[train_size+sliding-1:length-1]
	pred = []

	for i in range(len(testY)):
		pred.append(1)
	print mean_absolute_error(testY,pred)