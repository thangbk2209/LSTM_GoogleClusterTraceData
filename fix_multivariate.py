import pandas as pd
import numpy as np

from utils.GraphUtil import *
from utils.SlidingWindowUtil import SlidingWindow
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_dataset(n_sliding_window = 4, metric="cpu_rate"):
#     n_sliding_window = 4
    scaler = MinMaxScaler()
    scale_dat = scaler.fit_transform(dat[metric].reshape(-1,1))
    dat_sliding = np.array(list(SlidingWindow(scale_dat, n_sliding_window)))
    X_train_size = int(len(dat_sliding)*0.7)
    # sliding = np.array(list(SlidingWindow(dat_sliding, n_sliding_window)))
    # sliding = np.array(dat_sliding, dtype=np.int32)
    X_train = dat_sliding[:X_train_size]
    y_train = scale_dat[n_sliding_window:X_train_size+n_sliding_window].reshape(-1,1)
    X_test = dat_sliding[X_train_size:]
    y_test = scale_dat[X_train_size+n_sliding_window-1:].reshape(-1,1)
    return X_train.reshape(-1,n_sliding_window), X_test.reshape(-1,n_sliding_window), y_train, y_test

if __name__ == "__main__":
	n_sliding_window = 4
	dat = pd.read_csv('sampling_617685_metric_10min_datetime.csv',index_col=0, parse_dates=True)
	# cpu_X_train, cpu_X_test, cpu_y_train, cpu_y_test = get_dataset(n_sliding_window,"cpu_rate")
	mem_X_train, mem_X_test, mem_y_train, mem_y_test = get_dataset(n_sliding_window,"mem_usage")
	# print cpu_X_train
    print mem_X_train
