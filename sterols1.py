"""
Created on 05.02.18.
Find a relationship between Sterols in an Ice-Core sample. Use these to predict
other factors (age, water temperature etc.)
"""


import warnings
warnings.simplefilter(action = "ignore")
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


import sklearn.linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, utils


class Sterols():

    def __init__(self):
        pass

    def LoadData(self):
        self.data = pd.read_csv('/Users/katenewton/desktop/sterols.csv')
        print(self.data)

    def random_forest_regression(self):
        """ Perform Random Forest Regression on the X, Y data.
            RF: Random Forest
            MRF: Multi Random Forest """


        x = self.data[['1.0', '2.0', '3.0']].as_matrix()
        y = self.data['4.0'].as_matrix()

        test_size = 0.2  # use 30% of the data to test the algorithm (i.e 70% to train)
        random_state = 42  # keep this constant to keep the results constant
        max_depth = 10

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)


        # 1. make an instance of the RF algorithm called 'regr_rf'
        # 2. train it on the training dataset
        # 3. make predcitions about new y data
        regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        regr_rf.fit(x_train, y_train)  # create the RF algorithm
        y_rf = regr_rf.predict(x_test)  # predict on new data with RF
        rf_test = regr_rf.score(x_test, y_test)  # how well has RF done:
        print 'RF Test: ', rf_test








if __name__ == '__main__':
    s = Sterols()
    s.LoadData()
    s.random_forest_regression()
