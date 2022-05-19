# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data

from tutorial import *
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Other imported lib
import random

def topix():
  df = pd.read_csv("../data/tpx100data.csv")
  df['Change'] = df['Change'].str.rstrip('%') 
  df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")
  print(df)
  print(df.shape)
  print(df['Date'][0].year)
  return df

# Example try
def news(df):
  path = "../data/"+df['Date'][1400].strftime("%Y")+"/"
  path = path +df['Date'][1400].strftime("%m")+"/"+df['Date'][1400].strftime("%d")
  path = path +"/"+df['Date'][1400].strftime("%Y-%m-%d")+".csv"
  df_ = pd.read_csv(path)
  return

def extract_news_light(df):
  df_news_light = 0
  
  data = []
  for i in range(2):
    k = random.randint(0,1407) # based on dataset
    path = "../data/"+df['Date'][k].strftime("%Y")+"/"
    path = path +df['Date'][k].strftime("%m")+"/"+df['Date'][k].strftime("%d")
    path = path +"/"+df['Date'][k].strftime("%Y-%m-%d")+".csv"
    data.append(label_news( path, df['Change'][k] ))
  


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
  df = df.drop(df.index[droplist])
  
  print(df)
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
# Classifier: https://scikit-learn.org/stable/modules/ensemble.html#forest
# Word Embedding: https://nlp.stanford.edu/software/GloVe-1.2.zip
# Just to implement 10 of them
def random_forest_topix():
    
  return

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

