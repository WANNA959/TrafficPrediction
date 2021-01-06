"""
Train the NN model.
"""
import sys
import _thread
import keras
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from tkinter import ttk, filedialog, dialog
import  os
import tkinter
import tkinter.messagebox

warnings.filterwarnings("ignore")

file_path1=""
file_path2=""
modelName = None


def train_model(model, X_train, y_train, name, config,lag,callBack):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[callBack]
    )

    model.save('model/' + name + '-' + str(lag) + '.h5')

def train_allDense_model(model, X_train, y_train, name, config,lag,callBack):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop",metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        callbacks = [callBack]
    )

    model.save('model/' + name + '-' + str(lag) + '.h5')

def main(argv):

    config = {"batch": 256, "epochs": 20}
    file1 = 'data/100211data/100211_all_train.csv'
    file2 = 'data/100211data/100211_all_test.csv'

    #得到不同lag的lstm model
    # for i in range(16,18,2):
    #     lag = i
    #     X_train, y_train, _, _, _ = process_data(file1, file2, lag)
    #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #     m = model.get_lstm([lag, 64, 64, 1])
    #     train_model(m, X_train, y_train, "lstm", config,lag)

    #得到全连接神经网络训练model(lag=12
    lag=12
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)
    m = model.get_AllDense([lag, 64, 64, 1])
    train_allDense_model(m, X_train, y_train , "AllDense" , config , lag)


lagIntStart = 0
lagIntEnd = 0


def start_train():
    config = {"batch": 256, "epochs": 10}
    file_path1=fileStr1.get()
    file_path2=fileStr2.get()
    print(lagIntStart.get())
    print(lagIntEnd.get())
    print(modelName.get())
    if file_path1=="" or file_path2=="":
        tkinter.messagebox.askokcancel(title='请选择文件~', message='请选择两个文件')
        return
    print("start_train")
    callBack = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print("epoch",epoch)
    )
    needLstm =modelName.get()=="lstm" or modelName.get()=="all"
    needAllDense = modelName.get()=="allDense" or modelName.get()=="all"
    # _thread.start_new_thread(show_progress,())
    if needLstm:
        for i in range(lagIntStart.get(),lagIntEnd.get(),2):
            lag = i
            X_train, y_train, _, _, _ = process_data(file_path1, file_path2, lag)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([lag, 64, 64, 1])
            train_model(m, X_train, y_train, "lstm", config,lag,callBack)
            #_thread.start_new_thread(train_model,(m, X_train, y_train, "lstm", config,lag,callBack))
            # train_model(m, X_train, y_train, "lstm", config,lag,callBack)
    if needAllDense:
        X_train, y_train, _, _, _ = process_data(file_path1, file_path2, lagIntStart.get())
        m = model.get_AllDense([lagIntStart.get(), 64, 64, 1])
        train_allDense_model(m, X_train, y_train, "AllDense", config, lagIntStart.get(),callBack)
        #_thread.start_new_thread(train_allDense_model,(m, X_train, y_train, "AllDense", config, lagIntStart.get(),callBack))
        # train_allDense_model(m, X_train, y_train, "AllDense", config, lag,callBack)
    tkinter.messagebox.askokcancel(title='ok~', message='训练完成，结果保存在model文件夹下')
    return

# file_path2 = tkinter.StringVar()

def open_file_train():
    '''
    打开文件
    :return:
    '''
    file_path1 = filedialog.askopenfilename(title=u'选择训练集', initialdir=(os.path.expanduser('./data/100211data/100211_weekend_test.csv')))
    fileStr1.set(file_path1)
    print('打开文件：', file_path1)

def open_file_test():
    '''
    打开文件
    :return:
    '''
    file_path2 = filedialog.askopenfilename(title=u'选择测试集', initialdir=(os.path.expanduser('./data/100211data/100211_weekend_test.csv')))
    fileStr2.set(file_path2)
    print('打开文件：', file_path2)



window = tkinter.Tk()
window.title('入口')  # 标题
window.geometry('800x600')  # 窗口尺寸
def runUI():
    global lagIntStart
    lagIntStart = tkinter.IntVar()
    lagIntStart.set(4)
    global lagIntEnd
    lagIntEnd = tkinter.IntVar()
    lagIntEnd.set(12)
    global modelName
    modelName = tkinter.StringVar()
    frmL1 =tkinter.Frame( width=200, height=100,bg='blue')
    frmL2 =tkinter.Frame(width=200,height=100, bg='white')
    frmM1 =tkinter.Frame(width=200, height=10, bg='white')
    frmM2 = tkinter.Frame(width=2000, height=10,bg='yellow')
    frmL1.grid(row=0, column=0,padx=1,pady=1)
    frmL2.grid(row=1, column=0)
    frmM1.grid(row=0,column=1)
    frmM2.grid(row=1,column=1)

    #lag按钮
    frm22 =tkinter.Frame()
    frm21 =tkinter.Frame()
    frm31 =tkinter.Frame()
    frm32 =tkinter.Frame()
    frm22.grid(row=2,column=1)
    frm21.grid(row=2,column=0)
    frm31.grid(row=3,column=0)
    frm32.grid(row=3,column=1)
    tkinter.Label(frm21,text='输入lag start').pack()
    tkinter.Entry(frm22, textvariable=lagIntStart,width=40).pack()
    tkinter.Label(frm31,text='输入lag start').pack()
    tkinter.Entry(frm32, textvariable=lagIntEnd,width=40).pack()

    #选择模型下拉框
    frm41 = tkinter.Frame()
    frm42 = tkinter.Frame()
    frm41.grid(row=4,column=0)
    frm42.grid(row=4,column=1,)
    tkinter.Label(frm41, text='训练方法',).pack()
    dropBopx = ttk.Combobox(frm42,width=30,textvariable=modelName,state='readonly')
    dropBopx ['value'] = ('all', 'lstm', 'allDense')
    dropBopx.pack()
    dropBopx.current(0)
    # 开始训练按钮
    frm51=tkinter.Frame(width=30,height=10)
    frm51.grid(row=5,column=0,columnspan=2)
    frm52=tkinter.Frame(width=30,height=10)
    frm52.grid(row=5,column=1)
    # frmLB.grid(row=2,pady=4, column=0)
    # frmRT.grid(row=0, column=1, rowspan=3, padx=2, pady=3)
    global fileStr1
    fileStr1= tkinter.StringVar()
    global fileStr2
    fileStr2=tkinter.StringVar()
    tkinter.Entry(frmM1, textvariable=fileStr1,width=40).pack()
    tkinter.Entry(frmM2, textvariable=fileStr2,width=40).pack()
    tkinter.Button(frmL1, text='打开训练集', width=18,bg='orange', command=open_file_train).pack()
    tkinter.Button(frmL2, text='打开测试集', width=18,bg='orange', command=open_file_test).pack()
    tkinter.Button(frm51, text='开始训练', width=20, height=2,bg='orange', command=start_train).pack()
    # tkinter.Button(frm31,text='开始训练',width=30,height =2,bg='orange',command = open_file_test).pack()
    window.mainloop()
if __name__ == '__main__':
    runUI()
    # main(sys.argv)
