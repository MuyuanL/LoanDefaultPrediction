# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:57:26 2020

@author: Xingshuo
"""

import pandas as pd
import numpy as np
import math
import random
# read the first nrows rows of training data
# to read the whole file set nrows = 0
# returns data and labels in type of array
def read_train_data(filename, nrows=0):
    X = pd.DataFrame()
    if nrows <= 0:
        for chunk in pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False, chunksize=10000):
            X = pd.concat([X, chunk], ignore_index=True)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    X = fill_nan(X)
    data = np.asarray(X[:, 1:-4], dtype=float)
    labels = np.asarray(X[:, -1], dtype=float)
    return data, labels
# read the first nrows rows of test data
# to read the whole file set nrows = 0
# returns data in type of array
def read_test_data(filename, nrows=0):
    if nrows <= 0:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    X = fill_nan(X)
    data = np.asarray(X[:, 1:-3], dtype=float)
    return data


# to fill in the NAN values with the average of that column
def fill_nan(data):
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])
    return data

def train(X, y, model = 'linear', hyperparamter = 0):
    # for training data X, y, applying 5-fold cross-validation
    k = 5
    rmse = []
    # divide them into training set (X1, y1) and test set(X2, y2)
    for i in range(k):
        n = len(X)
        T_list = range(math.floor(n * i / k), math.floor(n * (i + 1) / k))
        S_list = np.setdiff1d(range(n), T_list)
        X1 = X[S_list, :]
        y1 = y[S_list]
        X2 = X[T_list, :]
        y2 = y[T_list]

        # calls the corresponding function that trains the model using X1, y1
        # and evaluate the performance on X2, y2
        if model == 'linear':
            rmse.append(linear_reg_train(X1, y1, X2, y2))

    # return average rmse
    return sum(rmse) / k


# train the model using X_train, y_train
# test on X_test, y_test, returns rmse
def linear_reg_train(X_train, y_train, X_test, y_test):
    n, d = X_train.shape
    
    # TODO: implement training 
    theta = np.zeros((d, 1))
    loss=10
    step_size=0.01
    eps=0.1
    max_iters=100
    iter_count=0
    # batch gradient descent
    while (loss>eps and iter_count<max_iters):
        loss=0
        thetaerr=np.zeros((d,1))
        for i in range(n):
        
            pred_y= X_train[i,:].dot(theta)
            for j in range(d):
                thetaerr[j]+=(max(pred_y, 0)-y_train[i])*X_train[i][j]
        for j in range(d):
            theta[j]=theta[j]-step_size*thetaerr[j]/n
        for i in range(n):
            pred_y= X_train[i,:].dot(theta)
            loss+=0.5*error_calc(pred_y,y_train[i])
        iter_count+=1
    print("iter_count",iter_count)
    print("final loss",loss)
    
    # stochchastic gradient desenct
    #while (loss>eps and iter_count<max_iters):
    #    loss=0
    #    i=random.randint(0,n-1)
    #    pred_y= X_train[i,:].dot(theta)
    #    for j in range(d):
    #        theta[j]=theta[j]-step_size*(pred_y-y_train[i])*X_train[i][j]
    #    for i in range(n):
    #        pred_y= X_train[i,:].dot(theta)
    #        loss+=0.5*error_calc(pred_y,y_train[i])
    #    iter_count+=1
    #print("iter_count",iter_count)
    #print("final loss",loss)
    #print ('theta: ',theta )

    
    # with theta acquired from training, calculate rmse
    n, d = X_test.shape
    mse = 0
    for i in range(n):
        mse += error_calc(X_test[i, :].dot(theta), y_test[i])
    return math.sqrt(mse)


def error_calc(pred_label, actual_label):
    return (max(pred_label, 0) - actual_label) ** 2


if __name__ == '__main__':
    train_file_name = 'train_v2.csv'
    test_file_name = 'test_v2.csv'
    X, y = read_train_data(train_file_name, 1000) # read the first 10000 rows of training data
    X_test = read_test_data(test_file_name, 1000) # read the first 10000 rows of test data

    # experiment with different models / hyperparameters, observes rmse
    linear_rmse = train(X, y, model='linear')

