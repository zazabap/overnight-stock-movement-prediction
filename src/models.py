from topix import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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