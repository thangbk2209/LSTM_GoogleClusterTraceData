
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas.lib import Timestamp
from scipy import signal
from segmentation import segment, fit

def period_detect(df, fs=1440, segment_method = "topdownsegment"):
    #dau vao df theo dinh dang cua twitter
    #fs: tan so lay mau (sample per day)
    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    if not segment_method in ["slidingwindowsegment", "topdownsegment", "bottomupsegment"]:
        raise ValueError("segment_method must be slidingwindowsegment or topdownsegment or bottomupsegment")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError(("data must be a 2 column data.frame, with the"
                              "first column being a set of timestamps, and "
                              "the second coloumn being numeric values."))

    if list(df.columns.values) != ["timestamp", "value"]:
        df.columns = ["timestamp", "value"]

    data_value = df["value"]

    n = len(data_value)
    data_value_trans= data_value - data_value.mean()

    # mien tan so
    data_value_trans= data_value - data_value.mean()
    fs = 60*24
    f, Pxx_den = signal.periodogram(data_value, fs)
    # chon nguong 40 %
    threshold = 0.2 * np.max(Pxx_den);
    index_period_candidate = [i for i in range(1,Pxx_den.size-1) if ((Pxx_den[i] > threshold) and (Pxx_den[i] > Pxx_den[i+1]) and (Pxx_den[i] > Pxx_den[i-1]))]
    period_candidate = [f[i] for i in index_period_candidate if (f[i]<(n-1)/fs)]
    period_candidate_pxx = [Pxx_den[i] for i in index_period_candidate if (f[i]<(n-1)/fs)]

    t = {
        'period': period_candidate,
        'magnitude': period_candidate_pxx
    }
    print "Getting all candidates..."
    period_candidate_point = pd.DataFrame(t)
    period_candidate_point = period_candidate_point.nlargest(8,'magnitude')


    lag = range(0,n-1)
    autocorr = [np.correlate(data_value, np.roll(data_value, -i))[0] / data_value.size for i in lag]
    ACF_candidate = [autocorr[int(i*fs)] for i in period_candidate_point['period']]


    final_all_period = []
    print "Checking all candidates of period..."
    print "There are %s candidates"%(len(period_candidate_point['period']))
    for idx, period_temp in enumerate(period_candidate_point['period']):
        print "Candidate %s: %s"%(idx+1,period_temp)
        startpoint = (int)(period_temp * fs)
        temp = autocorr[startpoint]

        begin_frame = np.max([(startpoint - fs), 0])
        end_frame = np.min([startpoint + fs, len(autocorr)])

        max = np.max(autocorr[begin_frame:end_frame])
        min = np.min(autocorr[begin_frame:end_frame])
        tb =(max+min) / 2
        autocorr_normalize = (np.array(autocorr) - tb) / (max-min)
        max_error = 0.005
        segments = []
        try:
            if(segment_method=="slidingwindowsegment"):
                segments = segment.slidingwindowsegment(autocorr_normalize[begin_frame:end_frame], fit.regression, fit.sumsquared_error, max_error)
            if (segment_method == "topdownsegment"):
                segments = segment.topdownsegment(autocorr_normalize[begin_frame:end_frame], fit.regression,
                                                        fit.sumsquared_error, max_error)
            if (segment_method == "bottomupsegment"):
                segments = segment.bottomupsegment(autocorr_normalize[begin_frame:end_frame], fit.regression,
                                                        fit.sumsquared_error, max_error)

        except:
            pass

        if len(segments) < 3:
            continue
        #check xem co la hill ko
        # diem start point la 200 (trong khoang moi 401 diem dang xet)
        # tim doan seg cua diem nay
        # seg_index = 0
        for i in range(0,len(segments)):
            if startpoint-begin_frame < segments[i][2]:
                seg_index = i
                break
        if ((seg_index < 2) or (seg_index > len(segments) - 2)):
            continue
        dh_trai = (segments[seg_index][3] - segments[seg_index][1]) - (
        segments[seg_index - 1][3] - segments[seg_index - 1][1])
        dh_phai = (segments[seg_index + 1][3] - segments[seg_index + 1][1]) - (
        segments[seg_index][3] - segments[seg_index][1])

        if ((dh_phai < 0) and (dh_trai < 0)):  # diem nam tren hill tien hanh tim closest peak
            while (segments[seg_index][3] > segments[seg_index][1]):
                # di tu trai sang phai
                # khi nao ma dao ham con duong thi di tu trai sang phai
                seg_index = seg_index + 1
                if (seg_index > len(segments) - 2):
                    break
            while (segments[seg_index][3] < segments[seg_index][1]):
                # khi nao dao ham con am thi di tu phai sang trai
                seg_index = seg_index - 1
                if ((seg_index < 2)):
                    break
            if ((seg_index >= 2) and (seg_index <= len(segments) - 2)):
                final_period = segments[seg_index][2]
                final_all_period.append(final_period + begin_frame)
    print "Periods: %s"%final_all_period
    return final_all_period
    #tim duoc segment cua diem

# In[ ]:


