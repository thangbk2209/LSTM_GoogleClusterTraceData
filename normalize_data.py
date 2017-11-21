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

colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

dataset = df.values

# normalize the dataset
length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
RAM_normal = scaler.fit_transform(dataset.T[1])
CPU_normal = scaler.fit_transform(dataset.T[0])

data=[]
data.append(CPU_normal)
data.append(RAM_normal)
data = np.array(data)
data = data.T
dataDf = pd.DataFrame(data)
dataDf.to_csv('data/normal.csv', index=False, header=None)
CPU = scaler.inverse_transform(CPU_normal)
CPUDf = pd.DataFrame(CPU)
CPUDf.to_csv('data/CPU.csv', index=False, header=None)