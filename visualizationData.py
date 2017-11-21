import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
# colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 

Real_df = read_csv('/home/nguyen/LSTM_GoogleTraceData/data/deseasonal_Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames, engine='python')
RealDataset = Real_df['cpu_rate'].values
print RealDataset

ax = plt.subplot()
ax.plot(RealDataset,label="Actual")
plt.xlabel("TimeStamp")
plt.ylabel("CPU")
plt.legend()
# plt.savefig('3layers64-32-16/batchsize=%s.png'%(batch_size))
plt.show()

