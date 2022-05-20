# Not my code
# Jack Hui's implementation 
# I am just listening to what he mentioned


from random import shuffle
from cppyy import sizeof
from topix import *
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval


from pathlib import Path
import json
import sys

DATA_DIR = "../"
NEWS_DIR = DATA_DIR + "dataset/"
MODEL_DIR = DATA_DIR + "models/"


def clean_blank(text):
  if not isinstance(text,str):
    return None
  while "  " in text:
    return text.replace("  ", " ")

def shorten_body(body):
  body_list = body.split(" ")
  body_list = body_list[:80]
  return " ".join(body_list)

def shorten_headline(headline):
  headline_list = headline.split(" ")
  headline_list = headline_list[:20]
  return " ".join(headline_list)

def combine_headlinebody(headline,body):
  if type(headline) is not None and type(body) is None:
    return shorten_headline(str(headline))
  elif type(headline) is None and type(body) is not None:
    return shorten_body(str(body))
  elif type(headline) is None and type(body) is None:
    return np.nan
  else:
    return shorten_headline(str(headline)) + " : " + shorten_body(str(body))

# 0 positive 1 neutral 2 negative
def diff_to_category(diff,std):
  if diff > 2.0 * std:
    return [1]
  elif diff < (-1.0) * 2.0 * std:
    return [0]
  else:
    return 0

def random_forest_topix_xu(X_train, X_test, y_train, y_test):
  model = RandomForestClassifier(n_estimators = 10)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  print("Done!")

def main():
  pricenews_df = pd.read_json(Path(NEWS_DIR+"price_news_v3.json"))
  pricenews_df = pricenews_df.T
  new_df = pricenews_df.copy()
  new_df.columns = ['id','day','timestamp','headline','body','category1','category2','category3','price_diff_100','price_diff_500']
  new_df['price_diff_100'] = new_df['price_diff_100'].astype(float)
  new_df['price_diff_500'] = new_df['price_diff_500'].astype(float)
  backup_df = new_df.copy()
  new_df = backup_df
  print("len of new_df",len(new_df))

  new_df['input_text'] = new_df.apply(lambda row : combine_headlinebody(row['headline'], row['body']), axis = 1)
  std_100 = new_df['price_diff_100'].std()
  std_500 = new_df['price_diff_500'].std()
  new_df['movement_category_100'] = new_df.apply(lambda row : diff_to_category(row['price_diff_100'], std_100), axis = 1)
  new_df['movement_category_500'] = new_df.apply(lambda row : diff_to_category(row['price_diff_500'], std_500), axis = 1)
  new_df_100 = new_df.query('movement_category_100 != 0')
  new_df_500 = new_df.query('movement_category_500 != 0')
  new_df_100.loc[:,'flag'] = 100
  new_df_500.loc[:,'flag'] = 500
  list_100_aug = new_df_100['input_text'].tolist()
  list_100_label = new_df_100['movement_category_100'].tolist()
  list_500_aug = new_df_500['input_text'].tolist()
  list_500_label = new_df_500['movement_category_500'].tolist()
  print("len of augmentation", len(list_100_aug))

  list_100_aug3 = []
  list_100_aug5 = []
  list_500_aug3 = []
  list_500_aug5 = []
  assert len(list_100_aug) == len(list_100_label) and len(list_500_aug) == len(list_500_label)
  for i in range(len(list_100_aug)-2):
    if list_100_label[i][0] != list_100_label[i+2][0]:
      i += 2
      #print(i)
      continue
    else:
      list_100_aug3.append([i, str(list_100_aug[i]) + " [SEP] " + str(list_100_aug[i+1]) + " [SEP] " + str(list_100_aug[i+2]), list_100_label[i]])

  for i in range(len(list_500_aug)-2):
    if list_500_label[i][0] != list_500_label[i+2][0]:
      i += 2
      #print(i)
      continue
    else:
      list_500_aug3.append([i, str(list_500_aug[i]) + " [SEP] " + str(list_500_aug[i+1]) + " [SEP] " + str(list_500_aug[i+2]), list_500_label[i]])

  for i in range(len(list_100_aug)-4):
    if list_100_label[i][0] != list_100_label[i+4][0]:
      i += 4
      #print(i)
      continue
    else:
      list_100_aug5.append([i, str(list_100_aug[i]) + " [SEP] " + str(list_100_aug[i+1]) + " [SEP] " + str(list_100_aug[i+2]) + " [SEP] " + str(list_100_aug[i+3]) + " [SEP] " + str(list_100_aug[i+4]), list_100_label[i]])

  for i in range(len(list_500_aug)-4):
    if list_500_label[i][0] != list_500_label[i+4][0]:
      i += 4
      #print(i)
      continue
    else:
      list_500_aug5.append([i, str(list_500_aug[i]) + " [SEP] " + str(list_500_aug[i+1]) + " [SEP] " + str(list_500_aug[i+2]) + " [SEP] " + str(list_500_aug[i+3]) + " [SEP] " + str(list_500_aug[i+4]), list_500_label[i]])

  new_df_100_aug3 = pd.DataFrame(list_100_aug3, columns = ['key', 'input_text','movement_category_100'])
  new_df_500_aug3 = pd.DataFrame(list_500_aug3, columns = ['key', 'input_text','movement_category_500'])
  new_df_100_aug5 = pd.DataFrame(list_100_aug5, columns = ['key', 'input_text','movement_category_100'])
  new_df_500_aug5 = pd.DataFrame(list_500_aug5, columns = ['key', 'input_text','movement_category_500'])

  print(len(new_df_100_aug3),
  len(new_df_100_aug5),
  len(new_df_500_aug3),
  len(new_df_500_aug5))

  print(new_df_100)

  X, y = bert_encoding(new_df_100_aug3)
  X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2 )
  X_test, X_valid, y_test, y_valid = train_test_split(X, y, shuffle=False, test_size=0.5 )
  random_forest_topix(X_train, X_test, y_train, y_test)
  naive_bayes_topix(X_train, X_test, y_train, y_test)
  k_nearest_neighbor_topix(X_train, X_test, y_train, y_test)
  neural_network_topix(X_train, X_test, y_train, y_test)





  # #original data
  # train_df_100, valid_df_100 = train_test_split(new_df_100, test_size=0.2, shuffle=False)
  # valid_df_100, test_df_100 = train_test_split(valid_df_100, test_size=0.5, shuffle=False)
  # train_df_500, valid_df_500 = train_test_split(new_df_500, test_size=0.2, shuffle=False)
  # valid_df_500, test_df_500 = train_test_split(valid_df_500, test_size=0.5, shuffle=False)
  
  # # Augmented data
  # train_df_100, valid_df_100 = train_test_split(new_df_100_aug3, test_size=0.2, shuffle=False)
  # valid_df_100, test_df_100 = train_test_split(valid_df_100, test_size=0.5, shuffle=False)
  # train_df_500, valid_df_500 = train_test_split(new_df_500_aug3, test_size=0.2, shuffle=False)
  # valid_df_500, test_df_500 = train_test_split(valid_df_500, test_size=0.5, shuffle=False)



# Tutorial Citation
# https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk
def bert_encoding(df):
  print("Bert_encoding")

  tmp = np.zeros(len(df))
  #print(sizeof(tmp))
  for index, row in df.iterrows():
    tmp[index] = row["movement_category_100"][0] 
  
  df['labels'] = tmp.tolist()
  sentences = df.input_text.values
  sentences = ["[CLS] " + str(Headline) + " [SEP]" for Headline in sentences]
  labels = df.labels.values
  #df["movement_category_100"] = pd.to_numeric( df["movement_category_100"])
  print(type(sentences[0]))
  print(type(df["movement_category_100"][0][0]))

  # Bring BERT Models in 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

  MAX_LEN = 256
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  return input_ids, labels 



if __name__ == "__main__" :
  main()
