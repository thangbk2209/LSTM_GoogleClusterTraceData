import pandas as pd 
import matplotlib.pyplot as plt
from pandas import read_csv

dataset = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()
# print a