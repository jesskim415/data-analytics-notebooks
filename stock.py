#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:03:53 2019

@author: jesskim
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#set working directory

dataset = pd.read_csv('googl.csv')'

prices = dataset['close'].tolist() # y - dependant value  (predict_value)
initial = dataset['open'].tolist() # - independant value (variables)
prices = np.reshape(prices, (len(prices), 1)) # making this into 1d vector
initial = np.reshape(initial, (len(initial), 1))


#checking the correlation between open and close  by visualizing
dataset[['open']].plot() #having [['open']] - gives the line
plt.title('open Price')
plt.show()

dataset[['close']].plot()
plt.title('close Price')
plt.show()

#training and test set. 
X_train, X_test, y_train, y_test = train_test_split(
        initial, prices, test_size = 0.2, random_state =0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#training set 
print('Train-set /','R2 score:',r2_score(y_train,regressor.predict(X_train))) #scoring the data
plt.scatter(X_train, y_train, color='red', label= 'Actual Price') #plotting the initial datapoints
plt.plot(X_train, regressor.predict(X_train), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression price| Open vs. Close')
plt.xlabel('Prices')
plt.legend()
plt.show()


#Test set
print('Test-set/','R2 score:',r2_score(y_test,regressor.predict(X_test)))
plt.scatter(X_test, y_test, color='red', label= 'Actual Price') #plotting the initial datapoints
plt.plot(X_test, regressor.predict(X_test), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression price| Open vs. Close')
plt.legend()
plt.xlabel('Prices')
plt.show()




