# Author: Shiwen An
# Date: 2022/07/01
# Purpose: Implementing RNN and LSTM
# for practice purpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

# sin波にノイズを付与する
def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

f = toy_problem()


def make_dataset(low_data, n_prev=100):
    data, target = [], []
    maxlen = 25
    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])
    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target


#g -> 学習データ，h -> 学習ラベル
g, h = make_dataset(f)

# モデル構築

# 1つの学習データのStep数(今回は25)
length_of_sequence = g.shape[1]
print(length_of_sequence)
in_out_neurons = 1
n_hidden = 300

# Batch size pertains to the
# amount of training samples to consider
# at a time for updating your network weights.
model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
model.fit(g, h,
          batch_size=300,
          epochs=100,
          validation_split=0.1,
          callbacks=[early_stopping]
          )

# 予測
predicted = model.predict(g)


def main():
    plt.figure()
    plt.plot(range(25, len(predicted) + 25), predicted, color="r", label="predict_data")
    plt.plot(range(0, len(f)), f, color="b", label="row_data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
