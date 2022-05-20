# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data

from tutorial import *
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.ensemble import RandomForestRegressor # Training Regressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Other imported lib
import random
import time

def topix():
  df = pd.read_csv("../data/tpx100data.csv")
  df['Change'] = df['Change'].str.rstrip('%') 
  df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")
  print(df)
  print(df.shape)
  print(df['Date'][0].year)
  return df

def extract_news_light(df):
  df_news_light = 0
  
  data = []
  for i in range(10):
    k = i*14 # based on dataset
    path = "../data/"+df['Date'][k].strftime("%Y")+"/"
    path = path +df['Date'][k].strftime("%m")+"/"+df['Date'][k].strftime("%d")
    path = path +"/"+df['Date'][k].strftime("%Y-%m-%d")+".csv"
    data.append(label_news( path, df['Change'][k] ))
  
  result = pd.concat(data)
  bert_encoding(result)

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


def random_forest_topix(X_train, X_test, y_train, y_test):
  print("Random Forest Topix")
  model = RandomForestClassifier(n_estimators = 10)
  #regressor = RandomForestRegressor(n_estimators=5, random_state=0)
  #regressor.fit(X_train, y_train)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  print("Done!")

def neural_network_topix(X_train, X_test, y_train, y_test):
  model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  print("Done!")
  
def k_nearest_neighbor_topix(X_train, X_test, y_train, y_test):
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  print("Done!")

def naive_bayes_topix(X_train, X_test, y_train, y_test):
  model = GaussianNB()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  print("Done!")

def train_bert():
    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 32
    
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    

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

