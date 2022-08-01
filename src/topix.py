# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data
import matplotlib.pyplot as plt
import numpy

from models import *
from tutorial import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(7)


# https://datatofish.com/plot-dataframe-pandas/
# For better analyzing the Stock data
# https://pythoninoffice.com/draw-stock-chart-with-python/

def main():
    df = topix()
    # news(df)
    LSTMMultivariant(df)
    #LSTM_one(df)
    # extract_news_light(df)

# https://www.tensorflow.org/guide/keras/rnn
def LSTMMultivariant(df):
    for i in range(len(df)):
        df.Date[i] = df.Date[i].timestamp() * 1000000 / 1000000
    dataset = df
    print(dataset)
    groups = [0, 1, 2, 3, 4, 6]

    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(df.Date, df.Open)
    plt.title(dataset.columns[1], y=0.5, loc='right')
    plt.subplot(5, 1, 2)
    plt.plot(df.Date, df.High)
    plt.title(dataset.columns[2], y=0.5, loc='right')
    plt.subplot(5, 1, 3)
    plt.plot(df.Date, df.Low)
    plt.title(dataset.columns[3], y=0.5, loc='right')
    plt.subplot(5, 1, 4)
    plt.plot(df.Date, df.Price)
    plt.title(dataset.columns[4], y=0.5, loc='right')
    plt.subplot(5, 1, 5)
    plt.plot(df.Date, df.Change)
    plt.title(dataset.columns[6], y=0.5, loc='right')
    plt.show()

    dataset = dataset[['Open', 'Price', 'High', 'Low', 'Change']]
    RNNLSTM(dataset)
    dataset = dataset.values

    # It might be a good starting point
    # https://ctowardsdatascience.com/how-to-convert-pandas-dataframe-to-keras-rnn-and-back-to-pandas-for-multivariate-regression-dcc34c991df9


def RNNLSTM(df):
    print(df)

    y_col = 'Open'
    test_size = int(len(df) * 0.2)  # the test data will be 10% (0.1) of the entire data
    train = df.iloc[:-test_size, :].copy()
    # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_index,col_indexer] = value instead
    test = df.iloc[-test_size:, :].copy()
    X_train = train.drop(y_col,axis=1).copy()
    y_train = train[[y_col]].copy() # the double brakets here are to keep the y in a dataframe format, otherwise it will be pandas Series
    print(X_train.shape, y_train.shape)

    Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
    Xscaler.fit(X_train)
    scaled_X_train = Xscaler.transform(X_train)
    print(X_train.shape)
    Yscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler.fit(y_train)
    scaled_y_train = Yscaler.transform(y_train)
    print(scaled_y_train.shape)
    scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)
    print(scaled_y_train.shape)

    scaled_y_train = np.insert(scaled_y_train, 0, 0)
    scaled_y_train = np.delete(scaled_y_train, -1)

    n_input = 25 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
    n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
    b_size = 32 # Number of timeseries samples in each batch
    generator = keras.preprocessing.sequence.TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)

    print(generator[0][0].shape)

    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    model.fit_generator(generator, epochs=5)

    X_test = test.drop(y_col, axis=1).copy()
    scaled_X_test = Xscaler.transform(X_test)
    test_generator = keras.preprocessing.sequence.TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)
    print(test_generator[0][0].shape)

    y_pred_scaled = model.predict(test_generator)
    y_pred = Yscaler.inverse_transform(y_pred_scaled)
    results = pd.DataFrame({'y_true': test[y_col].values[n_input:], 'y_pred': y_pred.ravel()})
    print(results)

    y1 = results.y_true.to_numpy()
    y2 = results.y_pred.to_numpy()
    print(y1)
    print(y2)
    x = numpy.arange(256)
    print(x)
    plt.plot(x, y1, 'g--')
    plt.plot(x, y2, 'b--')
    plt.show()

    score = model.evaluate(scaled_X_test, y_pred_scaled, verbose=0)
    print("Test Loss: ", score[0] )
    print("Test Accuracy: ", score[1])


    print("Everything Done")



# Data Augmentation in LSTM
# Could also be consider
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# Normal LSTM without language processing
def LSTM_one(df):
    dataset = df[['Open', 'Change']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))


def topix():
    df = pd.read_csv("../data/tpx100data.csv")
    df['Change'] = df['Change'].str.rstrip('%')
    df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")
    print(df)
    print(df.shape)
    print(df['Date'][0].year)
    df = df[::-1]
    # df = df.head(200)
    print(df)
    # remove all comma
    df.replace(',', '', regex=True, inplace=True)
    x = df.Date
    df.Open = pd.to_numeric(df['Open'])
    df.High = pd.to_numeric(df['High'])
    df.Price = pd.to_numeric(df['Price'])
    df.Low = pd.to_numeric(df['Low'])
    df.Change = pd.to_numeric(df.Change)
    # plt.yticks(np.arange(-100, 100, step=20))
    y = df.Open
    plt.plot(x, y, linewidth=2)
    plt.show()
    # Way to change ticks
    # fig = go.Figure(data=go.Scatter(x, y, mode='lines'))
    # fig.show()
    return df


def extract_news_light(df):
    df_news_light = 0

    data = []
    Default = 185
    for i in range(100):
        k = Default + i  # based on dataset
        path = "../data/" + df['Date'][k].strftime("%Y") + "/"
        path = path + df['Date'][k].strftime("%m") + "/" + df['Date'][k].strftime("%d")
        path = path + "/" + df['Date'][k].strftime("%Y-%m-%d") + ".csv"
        print(path)
        data.append(label_news(path, df['Change'][k]))

    result = pd.concat(data)
    print(result)
    # bert_encoding(result)


# Tutorial Citation
# https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk
def bert_encoding(data):
    df = data
    sentences = df.Headline.values
    sentences = ["[CLS] " + str(Headline) + " [SEP]" for Headline in sentences]
    labels = df.labels.values

    # Bring BERT Models in 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    MAX_LEN = 256
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    # Create a mask of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(input_ids, labels,
                                                                            shuffle=False, test_size=0.2)
    masks_train, masks_test, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=2022, test_size=0.5)

    # From here apply the random forest
    random_forest_topix(inputs_train, inputs_test, labels_train, labels_test)
    print("==================================================")
    neural_network_topix(inputs_train, inputs_test, labels_train, labels_test)
    print("==================================================")
    k_nearest_neighbor_topix(inputs_train, inputs_test, labels_train, labels_test)
    print("==================================================")
    naive_bayes_topix(inputs_train, inputs_test, labels_train, labels_test)


# label all the news
# based on increment or decrement
def label_news(path, label):
    print("label is a float", label)
    df = pd.read_csv(path)

    # Find out 9am basically
    droplist = []
    for index, row in df.iterrows():
        t = row['Story Date Time']
        dt = df.datetime.strptime(t, '%m/%d/%Y %H:%M:%S')
        r1 = isNowInTimePeriod(dt.time(9, 0), dt.time(15, 0), t.time())
        r2 = (t.weekday() < 5)
        # print("within market time", r1)
        # print("weekday()<5", r2)
        if r1 and r2: droplist.append(index)

    df = df[['Headline']]
    df = df.drop(df.index[droplist])

    if float(label) >= 0:
        labels = np.ones(len(df))
    else:
        labels = np.zeros(len(df))
    print("length of labels: ", len(labels))
    df['labels'] = labels.tolist()
    print(df)
    return df


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        # Over midnight:
        return nowTime >= startTime or nowTime <= endTime


# Reference
# https://scikit-learn.org/stable/modules/naive bayes.html#gaussian-naive-bayes 
def naive_bayes_topix():
    return


# https://scikit-learn.org/stable/modules/linear model.html#logistic-regression
def linear_regression_topix():
    return


# dismissed now
def han_topix():
    return


if __name__ == "__main__":
    main()
