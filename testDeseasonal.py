

import numpy as np
import pandas as pd
from pandas import Series, read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import add_trend
# trendData = read_csv('data/trend.csv')
# trend = trendData.values.T[0]

# seasonalData = read_csv('data/seasonal.csv')
# seasonal = seasonalData.values.T[0]

# residData = read_csv('data/resid.csv')
# resid = residData.values.T[0]
dat = Series.from_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes_todeseasonal.csv', header=0)
date_range = pd.date_range(dat.index.min(),dat.index.max(),freq="10min")
result = add_trend(dat.values)
# lol = np.add(trend,seasonal,resid)
# lolDf = pd.DataFrame(np.array(lol))
# lolDf.to_csv('data/lol.csv', index=False, header=None)
pyplot.plot(result)