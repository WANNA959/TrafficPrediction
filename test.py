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

import tkinter
from tkinter import ttk, filedialog, dialog
import os

from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

window = tkinter.Tk()
window.title('入口')  # 标题
window.geometry('400x600')  # 窗口尺寸

file_path=''

xls_text = tkinter.StringVar()
xls_text2 = tkinter.StringVar()
xls_text3 = tkinter.StringVar()

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
    return [vs,mape,mae,mse]

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

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data',linewidth=1)
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name,linewidth=1)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Volumns')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()#斜放

    plt.savefig("images/pre_weekend_time.png")
    # plt.show()

def open_file():
    '''
    打开文件
    :return:
    '''
    global file_path
    file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser('/Users/bytedance/python/trafficPrediction/data/100211data/100211_weekend_test.csv')))
    print('打开文件：', file_path)
    xls_text.set(file_path)


def compareMLPAndLSTM():
    lag = 10
    lstm = load_model("model/lstm-" + str(lag) + ".h5")
    allDense = load_model('model/AllDense-10.h5')
    models = [allDense,lstm]
    names = ['AllDense','lstm']
    file1 = 'data/100211data/100211_weekend_train.csv'
    file2 = 'data/100211data/100211_weekend_test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file_path, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    columnData=[]
    for name, model in zip(names, models):
        if name == 'lstm':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[0:288])
        print(name)
        evaValue = eva_regress(y_test, predicted)
        columnData.append(evaValue)
    plot_results(y_test[0:288], y_preds, names)
    print(columnData)
    print(np.array(columnData)[:,0])
    dataTable = tkinter.Toplevel()
    dataTable.title("MLP和LSTM的训练参数比较")
    dataTable.geometry("1600x1600")

    # 创建表格
    tree_date = ttk.Treeview(dataTable)

    # 定义列
    tree_date['columns'] = ['MLP', 'LSTM']
    tree_date.pack()

    # 设置列宽度
    tree_date.column('MLP', width=200)
    tree_date.column('LSTM', width=200)

    tree_date.heading('MLP', text='MLP')
    tree_date.heading('LSTM', text='LSTM')

    # 给表格中添加数据
    tree_date.insert('', 0, text='EVS', values=tuple(np.array(columnData)[:,0]))
    tree_date.insert('', 1, text='MAPE', values=tuple(np.array(columnData)[:,1]))
    tree_date.insert('', 2, text='MAE', values=tuple(np.array(columnData)[:,2]))
    tree_date.insert('', 3, text='MSE', values=tuple(np.array(columnData)[:,3]))

    img=Image.open('/Users/bytedance/python/trafficPrediction/images/pre_weekend_time.png')
    img_png = ImageTk.PhotoImage(img)
    label_img = ttk.Label(dataTable, image=img_png)
    label_img.pack()

    dataTable.mainloop()

def compareLSTMWithLag():
    file1 = 'data/100211data/100211_weekend_train.csv'
    file2 = 'data/100211data/100211_weekend_test.csv'

    dataTable = tkinter.Toplevel()
    dataTable.title("LSTM选取不同lag的训练参数比较")
    dataTable.geometry("1600x1600")
    # 创建表格
    tree_date = ttk.Treeview(dataTable)

    y_preds = []
    columnData=[]
    names=[]
    begin=xls_text2.get()
    end=xls_text3.get()
    begin=int(begin)
    end=int(end)
    columns=[]
    for lag in range(begin,end,2):
        columns.append('lag=' + str(lag))
        model = load_model("model/lstm-" + str(lag) + ".h5")
        _, _, X_test, y_test, scaler = process_data(file1, file_path, lag)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/lstm.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[0:288])
        print("lag=",lag)
        names.append("lstm lag="+str(lag))
        evaValue = eva_regress(y_test, predicted)
        columnData.append(evaValue)
    plot_results(y_test[0:288], y_preds, names)
    print(columnData)
    print(np.array(columnData)[:,0])

    #根据输入的lag范围动态生成表格
    tree_date['columns'] = columns
    for lag in range(begin,end,2):
        tree_date.column('lag=' + str(lag), width=200)
        tree_date.heading('lag=' + str(lag), text='lag=' + str(lag))
    # 定义列
    # tree_date['columns'] = ['lag=4','lag=6','lag=8','lag=10','lag=12','lag=14']
    tree_date.pack()

    # 设置列宽度
    # tree_date.column('lag=4', width=200)
    # tree_date.column('lag=6', width=200)
    # tree_date.column('lag=8', width=200)
    # tree_date.column('lag=10', width=200)
    # tree_date.column('lag=12', width=200)
    # tree_date.column('lag=14', width=200)
    #
    # tree_date.heading('lag=4', text='lag=4')
    # tree_date.heading('lag=6', text='lag=6')
    # tree_date.heading('lag=8', text='lag=8')
    # tree_date.heading('lag=10', text='lag=10')
    # tree_date.heading('lag=12', text='lag=12')
    # tree_date.heading('lag=14', text='lag=14')

    # 给表格中添加数据
    tree_date.insert('', 0, text='EVS', values=tuple(np.array(columnData)[:,0]))
    tree_date.insert('', 1, text='MAPE', values=tuple(np.array(columnData)[:,1]))
    tree_date.insert('', 2, text='MAE', values=tuple(np.array(columnData)[:,2]))
    tree_date.insert('', 3, text='MSE', values=tuple(np.array(columnData)[:,3]))

    img=Image.open('/Users/bytedance/python/trafficPrediction/images/pre_weekend_time.png')
    img_png = ImageTk.PhotoImage(img)
    label_img = ttk.Label(dataTable, image=img_png)
    label_img.pack()

    dataTable.mainloop()

def main():
    # lag = 12
    # lstm = load_model("model/lstm-"+str(lag)+".h5")
    # allDense = load_model('model/AllDense-12.h5')
    # models = [lstm]
    # models = [allDense]
    # names = ['lstm']
    # names = ['AllDense']
    # file1 = 'data/100211data/100211_weekend_train.csv'
    # file2 = 'data/100211data/100211_weekend_test.csv'
    # _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    #
    # y_preds = []
    # for name, model in zip(names, models):
    #     if name=='lstm':
    #         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #     file = 'images/' + name + '.png'
    #     plot_model(model, to_file=file, show_shapes=True)
    #     predicted = model.predict(X_test)
    #     predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    #     y_preds.append(predicted[0:288])
    #     print(name)
    #     eva_regress(y_test, predicted)
    #     # print(y_test.shape,predicted.shape)
    #
    # plot_results(y_test[0:288], y_preds, names)

    #入口
    padx=110#按钮居中
    #选择测试文件
    frame1 = tkinter.Frame(window)
    frame1.grid(row=0, column=0, sticky='w')
    bt3 = tkinter.Button(frame1, text='打开文件', width=18, height=2,bg='orange', command=open_file)
    bt3.pack(padx=padx)

    frame2 = tkinter.Frame(window)
    frame2.grid(row=1, column=0, sticky='w')
    xls = tkinter.Entry(frame2, textvariable=xls_text,width=100)
    xls_text.set(" ")
    xls.pack()
    # 对比MLP和LSTM
    frame3 = tkinter.Frame(window)
    frame3.grid(row=2, column=0, sticky='w')
    bt1 = tkinter.Button(frame3, text='对比MLP和LSTM', width=18, height=5, command=compareMLPAndLSTM)
    bt1.pack(padx=padx)

    fram4=tkinter.Frame(window)
    fram4.grid(row=3,column=0,sticky='w')
    tkinter.Label(fram4,text="lag:").pack(side='left')
    xls2 = tkinter.Entry(fram4, textvariable=xls_text2, width=20)
    xls2.pack(side='left',padx=5)
    tkinter.Label(fram4,text="~").pack(side='left')
    xls3 = tkinter.Entry(fram4, textvariable=xls_text3, width=20)
    xls_text2.set(" ")
    xls_text3.set(" ")
    xls3.pack(side='left')
    # 对比不同的lag
    frame5 = tkinter.Frame(window)
    frame5.grid(row=4, column=0,sticky='w')
    bt2 = tkinter.Button(frame5, text='对比LSTM不同的Lag', width=18, height=5, command=compareLSTMWithLag)
    bt2.pack(padx=padx)

    window.mainloop()



if __name__ == '__main__':
    main()
    # compareMLPAndLSTM()
