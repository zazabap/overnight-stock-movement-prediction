# Date: 2022/07/19
# Author: Chunying Quan
# Purpose: Basic Statistics for crime data
#          analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn import linear_model

from scipy.optimize import curve_fit

def read_csv():
  df = pd.read_csv("data/crime_data.csv")
  df.replace(',', '', regex=True,inplace=True)

  x = pd.to_numeric(df['crime'])
  y1 = pd.to_numeric(df['bullying'])
  y2 = pd.to_numeric(df['police'])

  print("======Minimum=====")
  print("Crime Min: ", min(x))
  print("Bullying Min ", min(y1))
  print("Police Min: ", min(y2))
  print("======Maximum=====")
  print("Crime Max: ", max(x))
  print("Bullying Max: ", max(y1))
  print("Police Max: ", max(y2))
  print("======Sum=====")
  print("Crime Sum: ", sum(x))
  print("Bullying Sum: ", sum(y1))
  print("Police Sum: ", sum(y2))
  print("======Average=====")
  print("Crime Avg: ", np.average(x))
  print("Bullying Avg: ", np.average(y1))
  print("Police Avg: ", np.average(y2))
  print("======Standard Deviation=====")
  print("Crime Std: ", np.std(x))
  print("Bullying Std: ", np.std(y1))
  print("Police Std: ", np.std(y2))
  print("======Median=====")
  print("Crime Median: ", np.median(x))
  print("Bullying Median: ", np.median(y1))
  print("Police Median: ", np.median(y2))  

  # plt.scatter(x, y1)
  plt.title("Crime vs. Police/Bullying Correlation")
  plt.xlabel("crime")
  plt.ylabel("Police/Bully")

  plt.scatter(x, y2, c ="pink",
            linewidths = 1,
            marker ="s",
            edgecolor ="green",
            s = 10)

  
  plt.scatter(x, y1, c ="yellow",
            linewidths = 1,
            marker ="^",
            edgecolor ="red",
            s = 10)


  X = np.array(x).reshape(-1,1)
  reg = linear_model.LinearRegression()
  reg.fit(X, y2)
  print("Coefficient of the model: ")
  print(reg.coef_)

  # plt.plot(x, y2, color="blue", linewidth=3)

  # y_fit = func(x, 0.32,0)
  # rng = np.random.default_rng()

  popt, pcov = curve_fit(func2, x, y1)
  popt2, pcov2 = curve_fit(func2, x, y2)

  z = np.polyfit(x, y2, 3)
  print("Fitting Parameter: ", z)
  f = np.poly1d(z)
  x_new = np.linspace(min(x), max(x), 100)  
  y_new = f(x_new)
  plt.plot(x_new, y_new, 'b-', linewidth=0.5, 
  label='Crime Police fitting')
  print("Fitting Parameter: ", z)
  z = np.polyfit(x, y1, 3)
  f = np.poly1d(z)
  x_new = np.linspace(min(x), max(x), 50)  
  y_new = f(x_new)
  plt.plot(x_new, y_new, 'g-', linewidth=0.5, 
  label='Bully Police fitting')

  # Log fitting y = a*log(x)+b
  z = np.polyfit(np.log(x), y2, 1)
  print("Fitting Parameter: ", z)
  f = np.poly1d(z)
  x_new = np.linspace(min(x), max(x), 100)  
  y_new = f(np.log(x_new))
  plt.plot(x_new, y_new, 'b--', linewidth=0.5, 
  label='Crime Police fitting Log')

  z =np.polyfit(np.log(x), y1, 1)
  print("Fitting Parameter: ", z)
  f = np.poly1d(z)
  x_new = np.linspace(min(x), max(x), 100)  
  y_new = f(np.log(x_new))
  plt.plot(x_new, y_new, 'g--', linewidth=0.5, 
  label='Bully Police fitting Log')

  # plt.plot(x, func2(x, *popt), 'b-', linewidth=0.5, 
  # label='Crime Police fitting')

  # plt.plot(x,func2(x, *popt2) , 'g-', linewidth=0.5, 
  # label='Crime Bullying fitting')

  plt.legend([ "Police Crime Fitting Poly(3)", "Bully Crime Fitting Poly(3)",
   "Police Crime Fitting Log", "Bully Crime Fitting Log",
   "Police", "Bullying" ])
  print("Stats for fitting: ")
  print(popt2)
  plt.show()

def func(x, a, b):
  return a*x+b

def func2(x, a, b, c):
  return a*x*x+b*x+c

if __name__ == '__main__':
  read_csv()
