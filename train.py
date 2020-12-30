"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config,lag):
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
        validation_split=0.05)

    model.save('model/' + name + '-'+str(lag)+'.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + '-'+str(lag)+' loss.csv', encoding='utf-8', index=False)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    config = {"batch": 256, "epochs": 600}
    file1 = 'data/100211data/100211_all_train.csv'
    file2 = 'data/100211data/100211_all_test.csv'
    for i in range(14,16,2):
        lag = i
        X_train, y_train, _, _, _ = process_data(file1, file2, lag)
        if args.model == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config,lag)
        if args.model == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config,lag)


if __name__ == '__main__':
    main(sys.argv)
