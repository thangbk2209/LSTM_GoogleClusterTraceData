import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['tims_stamp','cpu','mem','disk_io_time','disk_space'] 
# colnames = ['cpu','mem','disk_io_time','disk_space'] 
batch_size_array = [8,16,32,64,128]
realFile = ['/home/nguyen/LSTM_GoogleTraceData/data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv',
			'/home/nguyen/LSTM_GoogleTraceData/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin_todeseasonal.csv']

testFolder = ['/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/2layer_32-4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/mem/2layer_32-4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data5minutes/multivariate/cpu/2layer_32-4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/univariate/cpu/1layer_4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/univariate/mem/1layer_4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/1layer_4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/1layer_1neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/mem/1layer_4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/1layer_4neu_timeSteps/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data5minutes/univariate/cpu/1layer_4neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/1layer_4neu_relu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/2layer_4-2neu_timeSteps/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data5minutes/multivariate/cpu/2layer_32-4neu_timeSteps/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data5minutes/multivariate/cpu/2layer_32-4neu_timeSteps_deseasonal/',
			'/home/nguyen/LSTM_GoogleTraceData/results/notDecompose/data10minutes/multivariate/cpu/fix_1layer_1neu/',
			'/home/nguyen/LSTM_GoogleTraceData/results/Decompose/data10minutes/multivariate/cpu/2layer_32-4neu_timeSteps/']

sliding_widow = [2,3,4,5]
for sliding in sliding_widow:
	for batch_size in batch_size_array:
		# Real_df = read_csv('/home/hunter/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
		# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/LSTM_GoogleClusterTraceData/data/data_resource_JobId_6336594489.csv', header=None, index_col=False, names=colnames, usecols=[4], engine='python')
		Real_df = read_csv('%s'%(realFile[1]), header=None, index_col=False, names=colnames, engine='python')
		Pred_TestInverse_df = read_csv('%stestPredictInverse_sliding=%s_batchsize=%s.csv'%(testFolder[7],sliding,batch_size), header=None, index_col=False, engine='python')
		# Pred_Test_df = read_csv('/home/nguyen/LSTM_GoogleClusterTraceData/results/1layer_512neu/hehetestPredict_batchsize=%s.csv'%(batch_size), header=None, index_col=False, engine='python')
		RMSE_df = read_csv('%serror_sliding=%s_batchsize=%s.csv'%(testFolder[7],sliding,batch_size), header=None, index_col=False, engine='python')
		RealDataset = Real_df['mem'].values
		# train_size = int(len(RealDataset)*0.67)
		train_size = 2880
		test_size = len(RealDataset) - train_size
		print RealDataset
		Pred_TestInverse = Pred_TestInverse_df.values

		# predictions = []
		# for i in range(100):
		# 	predictions.append(Pred_TestInverse[i])
		# TestPredDataset = Pred_Test_df.values
		RMSE = RMSE_df.values[0][0]
		MAE = RMSE_df.values[1][0]
		print RMSE
		realTestData = []
		# for j in range(train_size+sliding, len(RealDataset),1):
		for j in range(train_size+sliding, len(RealDataset),1):
			realTestData.append(RealDataset[j])
		print len(realTestData)
		print len(Pred_TestInverse)
		# testScoreMAE = mean_absolute_error(Pred_TestInverse, realTestData)
		# print 'test score', testScoreMAE
		ax = plt.subplot()
		ax.plot(realTestData,label="Actual")
		ax.plot(Pred_TestInverse,label="predictions")
		# ax.plrot(TestPred,label="Test")
		plt.xlabel("TimeStamp")
		plt.ylabel("Mem")
		ax.text(0,0, 'testScore-sliding=%s-batch_size=%s: %s RMSE- %s MAE'%(sliding,batch_size, RMSE,MAE), style='italic',
		        bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
		plt.legend()
		# plt.savefig('mem5.png')
		plt.show()

