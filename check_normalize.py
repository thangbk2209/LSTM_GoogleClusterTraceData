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
df = read_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, engine='python')

# dataset = df.values

# normalize the dataset
# length = len(dataset)
meanCPU = np.mean(df['cpu_rate'].values)
maxCPU = np.max(df['cpu_rate'].values)
print meanCPU
print maxCPU
predictDf = read_csv('results/2layer_4-2neu/testPredict_batchsize=8.csv', header=None, index_col=False, names=['CPU'], engine='python')
# scaler = MinMaxScaler(feature_range=(0, 1))
CPU_pred = predictDf['CPU'].values
predict=[]
for cpu in CPU_pred:
	pred = cpu * maxCPU + meanCPU
	predict.append(pred)
print predict
# CPU_normal = scaler.fit_transform(dataset.T[0])
# RAM_normal = scaler.fit_transform(dataset.T[1])
CPUDf = pd.DataFrame(np.array(predict))
CPUDf.to_csv('results/2layer_4-2neu/aabbcc.csv', index=False, header=None)
