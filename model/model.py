"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))  # 防止过拟合
    model.add(Dense(units[3], activation='sigmoid'))  # 全连接层

    return model

def get_AllDense(units):
    """
    Build MLP Model with dense.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(Dense(units[1], input_shape=(units[0],), activation='sigmoid'))
    model.add(Dense(units[2], activation='sigmoid'))
    model.add(Dense(units[3], activation='sigmoid'))  # 全连接层

    return model
