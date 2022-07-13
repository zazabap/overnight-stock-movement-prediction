# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data

from models import *
from tutorial import *

# https://datatofish.com/plot-dataframe-pandas/
# For better analyzing the Stock data
# https://pythoninoffice.com/draw-stock-chart-with-python/

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
  df.replace(',', '', regex=True,inplace=True)
  x = df.Date
  y = pd.to_numeric( df['Open'])
  df.Change = pd.to_numeric( df.Change )
  #plt.yticks(np.arange(-100, 100, step=20))
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
    k = Default+i # based on dataset
    path = "../data/"+df['Date'][k].strftime("%Y")+"/"
    path = path +df['Date'][k].strftime("%m")+"/"+df['Date'][k].strftime("%d")
    path = path +"/"+df['Date'][k].strftime("%Y-%m-%d")+".csv"
    print(path)
    data.append(label_news( path, df['Change'][k] ))
  
  result = pd.concat(data)
  print(result)
  #bert_encoding(result)

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
      seq_mask = [float(i>0) for i in seq]
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
    t = dt.datetime.strptime( t, '%m/%d/%Y %H:%M:%S')
    r1 = isNowInTimePeriod(dt.time(9,0),dt.time(15,0), t.time())
    r2 = (t.weekday()<5)
    #print("within market time", r1)
    #print("weekday()<5", r2)
    if r1 and r2: droplist.append(index)

  df = df[['Headline']]
  df = df.drop(df.index[droplist])
  
  if float(label) >= 0 : labels = np.ones(len(df))
  else : labels = np.zeros(len(df))
  print("length of labels: ", len(labels) )
  df['labels'] = labels.tolist()
  print(df)
  return df


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        #Over midnight:
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


if __name__ == "__main__" :
  df = topix()
  #news(df)
  extract_news_light(df)