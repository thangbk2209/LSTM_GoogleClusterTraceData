
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adagrad, Adam, RMSprop
from keras.objectives import mean_squared_error
from keras.regularizers import l2
import seaborn as snb
from utils.GraphUtil import *
from utils.SlidingWindowUtil import SlidingWindow
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


dat = pd.read_csv('sampling_617685_metric_10min_datetime.csv', index_col=0, parse_dates=True)


# In[3]:


n_sliding_window = 4
scaler = MinMaxScaler()
scale_dat = scaler.fit_transform(dat['cpu_rate'])
dat_sliding = np.array(list(SlidingWindow(scale_dat, n_sliding_window)))
X_train_size = int(len(dat_sliding)*0.7)
# sliding = np.array(list(SlidingWindow(dat_sliding, n_sliding_window)))
# sliding = np.array(dat_sliding, dtype=np.int32)
X_train = dat_sliding[:X_train_size]
y_train = scale_dat[n_sliding_window:X_train_size+n_sliding_window].reshape(-1,1)
X_test = dat_sliding[X_train_size:]
y_test = scale_dat[X_train_size+n_sliding_window-1:].reshape(-1,1)




# # LSTM neural network

# In[22]:


from keras.layers import LSTM


# In[33]:


batch_size = 10
len_test = 1200
time_steps = 1
Xtrain = np.reshape(X_train, (X_train.shape[0], time_steps, n_sliding_window))
ytrain = np.reshape(y_train, (y_train.shape[0], time_steps, y_train.shape[1]))
Xtest = np.reshape(X_test, (X_test.shape[0], time_steps, n_sliding_window))


# In[43]:


model = Sequential()
model.add(LSTM(6,batch_input_shape=(batch_size,time_steps,n_sliding_window),stateful=False,activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[44]:


model.summary()


# In[ ]:


history = model.fit(Xtrain[:2900],y_train[:2900], nb_epoch=2000,batch_size=batch_size,shuffle=False, verbose=1,
                    validation_data=(Xtest[:len_test],y_test[:len_test]))


# In[ ]:


log = history.history
df = pd.DataFrame.from_dict(log)
get_ipython().magic(u'matplotlib')
df.plot(kind='line')


# In[62]:


y_pred = model.predict(Xtest,batch_size=batch_size)
# mean_absolute_error(y_pred,y_test[:len_test])


# In[63]:


y_pred


# In[51]:


get_ipython().magic(u'matplotlib')
plot_figure(y_pred=scaler.inverse_transform(y_pred), y_true=scaler.inverse_transform(y_test))


# In[52]:


results = []


# In[53]:


results.append({'score':mean_absolute_error(y_pred,y_test[:len_test]),'y_pred':y_pred})


# In[54]:


pd.DataFrame.from_dict(results).to_csv("lstm_result.csv",index=None)


# In[ ]:




