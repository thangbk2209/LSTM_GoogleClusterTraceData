import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# 
batch_size_array = [1,2,3,4,5,6,7,8,9,10]
for batch_size in batch_size_array:
# Real_df = read_csv('/home/hunter/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
	Real_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[1], engine='python')
	# Pred_Train_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/results/1layer_512neu/trainPredict.csv', header=None, index_col=False, engine='python')
	Pred_Test_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/results/3layers64-32-16/testPredict_batchsize=%s.csv'%(batch_size), header=None, index_col=False, engine='python')
	RMSE_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/results/3layers64-32-16/RMSE_batchsize=%s.csv'%(batch_size), header=None, index_col=False, engine='python')
	RealDataset = Real_df.values
	print RealDataset
	# TrainPredDataset = Pred_Train_df.values
	TestPredDataset = Pred_Test_df.values
	RMSE = RMSE_df.values[0][0]
	print RMSE
	TestPred = []
	for i in range(int(len(RealDataset)*0.67)):
		TestPred.append(np.nan)
	for i in range(len(TestPredDataset)):
		TestPred.append(TestPredDataset[i])

	ax = plt.subplot()
	ax.plot(RealDataset,label="Actual")
	# ax.plot(TrainPredDataset,label="Train")
	ax.plot(TestPred,label="Test")
	plt.xlabel("TimeStamp")
	plt.ylabel("CPU")
	ax.text(0,5, 'testScore-batch_size=%s: %s'%(batch_size, RMSE), style='italic',
	        bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
	plt.legend()
	plt.savefig('3layers64-32-16/batchsize=%s.png'%(batch_size))
	plt.show()

