import matplotlib as mpl
mpl.use('Agg')
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression as LR
import numpy as np
import pandas as pd 


from collections import Counter
# Counter(array1.most_common(1))
# print math.log(2,2)

# a =[1,2,3,4,5,1,2]
# print Counter(a.most_common(1))

def entro(X):
	x = [] # luu lai danh sach cac gia tri X[i] da tinh
	tong_so_lan = 0
	result = 0
	p=[]
	for i in range(len(X)):
		if Counter(x)[X[i]]==0:
			so_lan = Counter(X)[X[i]]
			tong_so_lan += so_lan
			x.append(X[i])
			P = 1.0*so_lan / len(X)
			p.append(P)
			result -= P * math.log(P,2)
		if tong_so_lan == len(X):
			break
	return result
def entroXY(X,Y):
	y = []
	result = 0
	pY = []
	tong_so_lan_Y = 0
	for i in range(len(Y)):
		# print Counter(y)[Y[i]]
		if Counter(y)[Y[i]]==0:
			x=[]
			so_lan_Y = Counter(Y)[Y[i]]
			tong_so_lan_Y += so_lan_Y
			y.append(Y[i])
			PY = 1.0* so_lan_Y / len(Y)
			# vi_tri = Y.index(Y[i])
			vi_tri=[]
			for k in range(len(Y)):
				if Y[k] == Y[i]: 
					vi_tri.append(k)
			for j in range(len(vi_tri)):
				x.append(X[vi_tri[j]])
			entro_thanh_phan = entro(x)
			result += PY * entro_thanh_phan
		if tong_so_lan_Y == len(Y):
			break
	return result
def infomation_gain(X,Y):
	return entro(X) - entroXY(X,Y)
def symmetrical_uncertainly(X,Y):
	return 2.0*infomation_gain(X,Y)/(entro(X)+entro(Y))


# colnames=['meanCPUUsage' ,'CMU' ,'AssignMem' ,'unmap_page_cache_memory_ussage' ,'page_cache_usage' ,'mean_local_disk_space', 'timeStamp']
# df = read_csv('/home/nguyen/spark-lab/spark-2.1.1-bin-hadoop2.7/google_cluster_analysis/results/my_offical_data_resource_TopJobId.csv', header=None, index_col=False, names=colnames)
colnames=['cpu_rate','mem_usage','disk_io_time','disk_space']
df = read_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames)

cpu_rate = df['cpu_rate'].values
mem_usage = df['mem_usage'].values
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values


su=[]
# entropyGGTrace = []
# # numberOfEntropy = 0
print symmetrical_uncertainly(cpu_rate,mem_usage)
# print symmetrical_uncertainly(cpu_rate,disk_io_time)
# print symmetrical_uncertainly(cpu_rate,disk_space)
# print symmetrical_uncertainly(disk_io_time,mem_usage)
# print symmetrical_uncertainly(disk_space,mem_usage)
# print symmetrical_uncertainly(disk_space,disk_io_time)

for i in range(len(colnames)):
	sui=[]
	for k in range(i+1):
		if(k==i):
			sui.append(1)
		else:
			sui.append(symmetrical_uncertainly(df[colnames[i]].values,df[colnames[k]].values))
	for j in range(i+1, len(colnames),1):
		sui.append(symmetrical_uncertainly(df[colnames[i]].values,df[colnames[j]].values))
	su.append(sui)
print su
# su=[[1,2,3],[2,3,4]]
dataFuzzyDf = pd.DataFrame(np.array(su))
dataFuzzyDf.to_csv('data/symetrical_uncertainty_data_resource_JobId_6336594489_5minutes.csv', index=False, header=None)
