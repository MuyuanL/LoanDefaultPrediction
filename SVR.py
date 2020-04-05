# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:35:35 2020

@author: Xingshuo
"""

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import liblinearutil
"""
To change to native model: Line 155 - 157
To change max iteration: Line 109
To change the way to fill in NaN value: Line 60
To change step size / precision: Line 155
"""


# read the first nrows rows of training data
# to read the whole file set nrows = 0
# returns data and labels in type of array
def read_train_data(filename, nrows=0,frac=1):
    X = pd.DataFrame()
    if nrows <= 0:
        for chunk in pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, chunksize=10000):
            X = pd.concat([X, chunk], axis=0, ignore_index=True)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    n,d= X.shape
    X=X[0:math.floor(frac*n/4),:]
    X = fill_nan(X)
    data = np.asarray(X[:, 1:-4], dtype=float)
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]
    # data = np.asarray(X[:, 1:5], dtype=float)  # just use figure 1-5 for basic testing for now
    # Normalize every column of data so that it ranges from 0 to 1
    data_normed = (data - data.min(0)) / data.ptp(0)
    labels = np.asarray(X[:, -1], dtype=float)
    labels = np.reshape(labels, (-1, 1))
    return data_normed, labels


# read the first nrows rows of test data
# to read the whole file set nrows = 0
# returns data in type of array
# (Deprecated)
def read_test_data(filename, nrows=0):
    """
    if nrows <= 0:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=True, error_bad_lines=True, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    X = fill_nan(X)
    data = np.asarray(X[:, 1:-3], dtype=float)
    return data
    """
    return None


# to fill in the NAN values
def fill_nan(data):
    col_mean = np.nanmean(data, axis=0)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_median = np.nanmedian(data, axis=0)
    use = col_min
    inds = np.where(np.isnan(data))
    data[inds] = np.take(use, inds[1])  # Fill in NaN with <use>
    # data[inds] = 0  # Fill in NaN with 0
    return data


# train using X, y, specified model name, and hyperparamter if appliable
# will return the calculated error
def train(X, y, model = 'linear', hyperparamter = 0):
    # for training data X, y, applying 5-fold cross-validation
    k = 5
    train_rmse = []
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
        print(f'======== {k}-fold: Iteration {i} ========')
       
        
        if model == 'svr':
           train_rmse_linear, test_rmse_linear = svr_train(X1, y1, X2, y2)
           train_rmse.append(train_rmse_linear)
           rmse.append(test_rmse_linear)

    # return average rmse
    avg_train_rmse = sum(train_rmse) / k
    avg_rmse = sum(rmse) / k
    print()
    print(f'++++ Result ++++')
    print(f'Model: {model}, average train rmse: {avg_train_rmse}, average test rmse: {avg_rmse}')
    print()
    return avg_rmse

def svr_train(X_train, y_train, X_test, y_test):
    n1, d1 = X_train.shape
    n2, d2 = X_test.shape
  
    #y1 = list(np.reshape(y_train,(1,-1)))
    #y2 = list(np.reshape(y_test,(1,-1)))
    y1=np.squeeze(y_train)
    y2=np.squeeze(y_test)
    
#    scaler = StandardScaler()
#    scaler=MinMaxScaler(feature_range=(0, 1))
#    X_train = scaler.fit_transform(X_train)
    modeltrained = liblinearutil.train(y1, X_train, '-s 11 -c 0.1')
    
#training rmse
    p_label, p_acc, p_val = liblinearutil.predict(y1, X_train, modeltrained)
    pred_y1 = np.array(p_label)
    yerr1 = np.zeros((1,n1))
    pred_y1 =np.maximum(pred_y1, np.zeros((1,n1))) 
    y_train=np.reshape(y_train,(1,-1))
    yerr1 =np.subtract(pred_y1,y_train)
    
    loss1 =np.sqrt(((yerr1) ** 2).mean())
    
#testing rmse
    p_label, p_acc, p_val = liblinearutil.predict(y2, X_test, modeltrained) 
    pred_y2 = np.array(p_label)
    yerr2 = np.zeros((1,n1))
    pred_y2 =np.maximum(pred_y2, np.zeros((1,n2))) 
    y_train=np.reshape(y_test,(1,-1))
    yerr2 =np.subtract(pred_y2,y_test)
    loss2 =np.sqrt(((yerr2) ** 2).mean())
    print("train rmse",loss1,"test rmse",loss2)
    return (loss1,loss2)



if __name__ == '__main__':
    train_file_name = 'train_v2.csv'
    # test_file_name = 'test_v2.csv'
    X1, y1 = read_train_data(train_file_name, 0,1) # read the first <num> rows of training data   
    linear_rmse = train(X1, y1, model='svr')
    X2, y2 = read_train_data(train_file_name, 0,2) # read the first <num> rows of training data   
    linear_rmse = train(X2, y2, model='svr')
    X3, y3 = read_train_data(train_file_name, 0,3) # read the first <num> rows of training data   
    linear_rmse = train(X3, y3, model='svr')
    X4, y4 = read_train_data(train_file_name, 0,4) # read the first <num> rows of training data   
    linear_rmse = train(X4, y4, model='svr')
#    X, y = read_train_data(train_file_name, 50000) # read the first <num> rows of training data   
#    linear_rmse = train(X, y, model='svr')
