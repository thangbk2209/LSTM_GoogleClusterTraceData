import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('international-airline-passengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse, infer_datetime_format=True)

# dat = Series.from_csv('data/international-airline-passengers.csv')
# print len(dat)
# date_range = pd.date_range(dat.index.min(),dat.index.max(),freq="Month")
print data
result = seasonal_decompose(data.values, model='additive',freq=12)
print result
trend = result.trend
seasonal = result.seasonal
resid = result.resid
print trend
print len(seasonal)
print resid
# arr = np.array(result.resid)
residDf = pd.DataFrame(np.array(resid))
residDf.to_csv('data/residair.csv', index=False, header=None)
trendDf = pd.DataFrame(np.array(trend))
trendDf.to_csv('data/trendair.csv', index=False, header=None)
seasonalDf = pd.DataFrame(np.array(seasonal))
seasonalDf.to_csv('data/seasonalair.csv', index=False, header=None)
# print arr
result.plot()
pyplot.show()
