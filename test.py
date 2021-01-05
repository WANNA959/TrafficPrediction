# -*- coding: utf-8 -*-
"""
Traffic Flow Prediction with Neural Networks(LSTM、GRU).
"""
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from data.data import process_data

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

# 评价参数
def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)#平均绝对百分比误差
    print('mae:%f' % mae)#平均绝对误差
    print('mse:%f' % mse)#均方根误差
    print('rmse:%f' % math.sqrt(mse))

# 真实数据和预测数据对比
def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2018-04-05 00:00'
    x = pd.date_range(d, periods=len(y_true),freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Volumns')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()#斜放

    plt.show()
    # plt.savefig("images/pre_weekend_time.png")

def main():
    lag = 16
    lstm = load_model("model/lstm-"+str(lag)+".h5")
    allDense = load_model('model/AllDense-10.h5')
    models = [lstm]
    # models = [allDense]
    names = ['lstm']
    # names = ['AllDense']
    file1 = 'data/100211data/100211_weekend_train.csv'
    file2 = 'data/100211data/100211_weekend_test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name=='lstm':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[0:288])
        print(name)
        eva_regress(y_test, predicted)
        # print(y_test.shape,predicted.shape)

    plot_results(y_test[0:288], y_preds, names)


if __name__ == '__main__':
    main()
