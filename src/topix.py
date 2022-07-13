# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data




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
    dataset = dataset.values
    print(dataset)
    # Use Price+High+Low+Change to predict the Open Price
    #encoder = LabelEncoder()
    #dataset[:, 0] = encoder.fit_transform(dataset[:, 0])
    print(dataset)
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    print(reframed)


    # split into train and test sets
    values = reframed.values
    n_train_days = 365
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(10))
    model.summary()

    # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dense(1))
    # model.compile(loss='mae', optimizer='adam')
    # # fit network
    # history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
    #                     shuffle=False)
    # # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()



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
        t = dt.datetime.strptime(t, '%m/%d/%Y %H:%M:%S')
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
