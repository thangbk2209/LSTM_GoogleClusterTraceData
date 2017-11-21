import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
def plot_figure(y_pred=None,y_true=None,color=['blue','red'],title=None):
    ax = plt.subplot()
    ax.set_color_cycle(color)
    ax.plot(y_pred,'--',label='Predict')
    ax.plot(y_true,label='Actual')
    ax.legend()
    ax.set_title(title)
    plt.show()
    return None
def plot_figure_with_label(y_pred=None,y_test=None,title=None,metric=None):
    ax = plt.subplot()
    ax.set_color_cycle(['blue','red'])
    ax.plot(y_pred,'--', label='Predict')
    ax.plot(y_test, label='Actual')
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('Time (minutes)')
    plt.legend()
    return ax

def plot_metric_figure(y_pred=None,y_test=None,metric_type=None,title=None):
    for k, metric in enumerate(metric_type):
        plot_figure(y_pred[:,k],y_test[:,k],title="%s based on %s Prediction - score %s"
                                                  %(metric,title,mean_squared_error(y_pred[:, k], y_test[:, k])))