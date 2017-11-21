import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
# colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
batch_size_array = [8,16,32,64,128]

sliding_widow = [1,2,3,4,5]
for sliding in sliding_widow:
	for batch_size in batch_size_array:
		# Real_df = read_csv('/home/hunter/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
		# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
		Real_df = read_csv('/home/nguyen/LSTM_GoogleTraceData/data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames, engine='python')
		Pred_TestInverse_df = read_csv('/home/nguyen/LSTM_GoogleTraceData/resultsdataDeseasonal5minutes/2layer_32-4neu/testPredictInverse_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), header=None, index_col=False, engine='python')
		# Pred_Test_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/results/1layer_512neu/hehetestPredict_batchsize=%s.csv'%(batch_size), header=None, index_col=False, engine='python')
		RMSE_df = read_csv('/home/nguyen/LSTM_GoogleTraceData/resultsdataDeseasonal5minutes/2layer_32-4neu/error_sliding=%s_batchsize=%s.csv'%(sliding,batch_size), header=None, index_col=False, engine='python')
		RealDataset = Real_df['cpu_rate'].values
		print RealDataset
		Pred_TestInverse = Pred_TestInverse_df.values
		# TestPredDataset = Pred_Test_df.values
		RMSE = RMSE_df.values[0][0]
		MAE = RMSE_df.values[1][0]
		print RMSE
		# TestPred = []
		# for i in range(int(len(RealDataset)*0.67)):
		# 	TestPred.append(np.nan)
		# for i in range(len(TestPredDataset)):
		# 	TestPred.append(TestPredDataset[i])

		TestPredInverse = []
		for i in range(int(len(RealDataset)*0.67)):
			TestPredInverse.append(np.nan)
		for i in range(len(Pred_TestInverse)):
			TestPredInverse.append(Pred_TestInverse[i])

		ax = plt.subplot()
		ax.plot(RealDataset,label="Actual")
		ax.plot(TestPredInverse,label="Predict Test")
		# ax.plot(TestPred,label="Test")
		plt.xlabel("TimeStamp")
		plt.ylabel("CPU")
		ax.text(0,50, 'testScore-sliding=%s-batch_size=%s: %s RMSE- %s MAE'%(sliding,batch_size, RMSE,MAE), style='italic',
		        bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
		plt.legend()
		# plt.savefig('3layers64-32-16/batchsize=%s.png'%(batch_size))
		plt.show()

