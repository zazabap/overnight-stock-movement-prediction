# Author: Shiwen An
# Date: 2022/05/18
# Purpose: Wash the data

from tutorial import *
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def topix():
  df = pd.read_csv("../data/tpx100data.csv")
  df['Change'] = df['Change'].str.rstrip('%') 
  df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")
  print(df)
  print(df.shape)
  print(df['Date'][0].year)
  return df

def news(df):
  path = "../data/"+df['Date'][1400].strftime("%Y")+"/"
  path = path +df['Date'][1400].strftime("%m")+"/"+df['Date'][1400].strftime("%d")
  path = path +"/"+df['Date'][1400].strftime("%Y-%m-%d")+".csv"
  df_ = pd.read_csv(path)
  return

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
  news(df)

