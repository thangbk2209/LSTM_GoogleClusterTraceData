import numpy as np
import pandas as pd
from pandas import Series,read_csv
from matplotlib import pyplot as plt
colnames = ['time_stamp', 'CPU']
original = read_csv('data/Fuzzy_CPU_sampling_617685_metric_10min_datetime_origin.csv',names = colnames)
originalData = original['CPU']
originalData = np.array(originalData)
seasonal = read_csv('data/seasonalCPU10minutes.csv',names=['CPU'])
seasonalData = seasonal['CPU']
seasonalData = np.array(seasonalData)


trend_na = read_csv('data/trendCPU10minutes.csv',names=['CPU'])
trend = trend_na.fillna(0)['CPU']

print seasonalData
print len(seasonalData)
print len(originalData)
print 'dsfads'
print len(trend)
deseasonal = []
for i in range(len(originalData)):
	deseasonal.append(originalData[i] - seasonalData[i]-trend[i])
print deseasonal

ax = plt.subplot()
ax.plot(originalData,label="Original")
ax.plot(deseasonal,label="deseasonal")
# ax.plot(TestPred,label="Test")
plt.xlabel("TimeStamp")
plt.ylabel("Mem")
plt.legend()
# plt.savefig('3layers64-32-16/batchsize=%s.png'%(batch_size))
plt.show()
deseasonalDf = pd.DataFrame(np.array(deseasonal))
deseasonalDf.to_csv('data/deseasonalCPU10minutes.csv', index=False, header=None)