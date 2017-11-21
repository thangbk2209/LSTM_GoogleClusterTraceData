import numpy as np
from sklearn.metrics import *
def load_file(name):
	dat = np.load(name)
	return dat['y_pred'],dat['y_true']
