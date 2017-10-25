import numpy as np 
import pandas as pd 
from pandas import read_csv

colnames=['time_stamp','taskIndex','machineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampling_portion','agg_type','sampled_cpu_usage']
df = read_csv('data/data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames, engine='python')


cpu_rate = df['meanCPUUsage'].values
mem_usage= df['AssignMem'].values
disk_io_time = df['mean_diskIO_time'].values
disk_space = df['mean_local_disk_space'].values
print 'lol'
print len(cpu_rate)
print cpu_rate
fuzzy_cpu_rate=[]
fuzzy_mem_usage=[]
fuzzy_disk_io_time=[]
fuzzy_disk_space=[]
for i in range(len(cpu_rate)):
	fuzzy_cpu_rate.append(round(cpu_rate[i],2))
	fuzzy_mem_usage.append(round(mem_usage[i],2))
	fuzzy_disk_io_time.append(round(disk_io_time[i],2))
	fuzzy_disk_space.append(round(disk_space[i],2))
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
dataFuzzyDf.to_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', index=False, header=None)