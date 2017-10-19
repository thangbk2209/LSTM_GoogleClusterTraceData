import numpy as np 
import pandas as pd 
from pandas import read_csv

colnames = ['number','time','jID','moment','cpu_rate','mem_usage','disk_io_time','disk_space']
df = read_csv('data/sampling_617685_metric_10min_datetime_origin.csv', names=colnames)

cpu_rate = df['cpu_rate'].values
mem_usage= df['mem_usage'].values
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values
print 'lol'
print len(cpu_rate)
print cpu_rate
fuzzy_cpu_rate=[]
fuzzy_mem_usage=[]
fuzzy_disk_io_time=[]
fuzzy_disk_space=[]
for i in range(len(cpu_rate)):
	fuzzy_cpu_rate.append(round(cpu_rate[i],3))
	fuzzy_mem_usage.append(round(mem_usage[i],3))
	fuzzy_disk_io_time.append(round(disk_io_time[i],3))
	fuzzy_disk_space.append(round(disk_space[i],3))
# print fuzzy_cpu_rate
# print fuzzy_mem_usage
# print fuzzy_disk_space
# print fuzzy_disk_io_time
dataFuzzy=[]
dataFuzzy.append(fuzzy_cpu_rate)
dataFuzzy.append(fuzzy_mem_usage)
dataFuzzy.append(fuzzy_disk_io_time)
dataFuzzy.append(fuzzy_disk_space)
dataFuzzyDf = pd.DataFrame(np.array(dataFuzzy).T)
dataFuzzyDf.to_csv('data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', index=False, header=None)