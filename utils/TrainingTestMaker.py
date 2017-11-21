import numpy as np
from sklearn.cross_validation import train_test_split
class TrainingTestMaker(object):
    def make_fuzzy_test(self, preprocess_data, target, real_target, train_size=0.7):
        X_train_size = int(len(real_target) * 0.7)
        sliding = np.array(preprocess_data, dtype=np.int32)
        sliding_number = len(preprocess_data[0])
        X_train = sliding[:X_train_size]
        y_train = target[sliding_number:X_train_size + sliding_number]
        X_test = sliding[X_train_size:]
        # y_test = target[X_train_size + sliding_number - 1:]
        y_actual_test = real_target[X_train_size + sliding_number - 1:].tolist()
        return X_train,y_train, X_test,y_actual_test
