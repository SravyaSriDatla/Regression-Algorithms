import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'/Users/bannusagi/Documents/emp_sal.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fitting svr(sigmoid) to dataset
from sklearn.svm import SVR
regressor_sigmoid = SVR(kernel='sigmoid', gamma='auto')
regressor_sigmoid.fit(X,y)

y_pred_svr_sigmoid = regressor_sigmoid.predict([[6.5]])

#fitting svr(polynomial)
from sklearn.svm import SVR
regressor_poly = SVR(kernel='poly',degree=4)
regressor_poly.fit(X,y)

y_pred_svr_poly = regressor_poly.predict([[6.5]])

#fitting svr(RBF) / default 
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)

y_pred_svr = regressor.predict([[6.5]])


#knn 
from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=5,p=2,weights='distance',algorithm='ball_tree')
regressor_knn.fit(X, y)

y_pred_knn1 = regressor_knn.predict([[6.5]])

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=5,p=2,weights='distance',algorithm='kd_tree')
regressor_knn.fit(X, y)

y_pred_knn2 = regressor_knn.predict([[6.5]])

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=5,p=1,weights='distance',algorithm='brute')
regressor_knn.fit(X, y)

y_pred_knn3 = regressor_knn.predict([[6.5]])

#tree
from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor()
regressor_tree.fit(X,y)

y_pred_tree= regressor_tree.predict([[6.5]])

from sklearn.tree import DecisionTreeRegressor
regressor_tree1 = DecisionTreeRegressor(criterion="absolute_error",splitter='random',max_depth=2,random_state=0)
regressor_tree1.fit(X,y)

y_pred_tree1= regressor_tree1.predict([[6.5]])

from sklearn.tree import DecisionTreeRegressor
regressor_tree2 = DecisionTreeRegressor(criterion="friedman_mse",splitter='random',max_depth=2,random_state=0)
regressor_tree2.fit(X,y)

y_pred_tree2= regressor_tree2.predict([[6.5]])

#ensemble
from sklearn.ensemble import RandomForestRegressor
regressor_random= RandomForestRegressor()
regressor_random.fit(X,y)

y_pred_random= regressor_random.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor
regressor_random1= RandomForestRegressor(n_estimators=30,criterion="absolute_error",max_depth=2,random_state=0)
regressor_random1.fit(X,y)

y_pred_random1= regressor_random.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor
regressor_random2= RandomForestRegressor(criterion="absolute_error",max_depth=2, n_estimators=50)
regressor_random2.fit(X,y)

y_pred_random2= regressor_random.predict([[6.5]])
