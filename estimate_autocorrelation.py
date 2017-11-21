from numpy import *
import numpy as np
import pylab as P
from matplotlib import pyplot as plt
from pandas import read_csv
# fn = 'data.txt'
# x = loadtxt(fn,unpack=True,usecols=[1])
# time = loadtxt(fn,unpack=True,usecols=[0]) 
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
# series = Series.from_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=0)
df = read_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames, engine='python')
CPU = df['cpu_rate'].values
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

plt.plot(estimated_autocorrelation(CPU))
plt.xlabel('time (s)')
plt.ylabel('autocorrelation')
plt.show()