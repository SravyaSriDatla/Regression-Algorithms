import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'/Users/bannusagi/Documents/emp_sal.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fitting svr to dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)

y_pred_svr = regressor.predict([[6.5]])