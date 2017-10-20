import numpy as np
import matplotlib
from time import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
import math
# import keras
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)

# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph/test.png', histogram_freq=0,  write_graph=True, write_images=True)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[1,2], engine='python')

dataset = df.values

# normalize the dataset
length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

CPU_normal = scaler.fit_transform(dataset.T[0])
RAM_normal = scaler.fit_transform(dataset.T[1])
CPUDf = pd.DataFrame(np.array(CPU_normal))
CPUDf.to_csv('data/CPU_normal.csv', index=False, header=None)
