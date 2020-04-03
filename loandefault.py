import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression


# read the first nrows rows of training data
# to read the whole file set nrows = 0
# returns data and labels in type of array
def read_train_data(filename, nrows=0):
    X = pd.DataFrame()
    if nrows <= 0:
        for chunk in pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, chunksize=10000):
            X = pd.concat([X, chunk], axis=0, ignore_index=True)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    X = fill_nan(X)
    data = np.asarray(X[:, 1:-4], dtype=float)
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]
    # data = np.asarray(X[:, 1:5], dtype=float)  # just use figure 1-5 for basic testing for now
    # Normalize every column of data so that it ranges from 0 to 1
    data_normed = (data - data.min(0)) / data.ptp(0)
    labels = np.asarray(X[:, -1], dtype=float)
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


# to fill in the NAN values with the average of that column
def fill_nan(data):
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])
    return data


# train using X, y, specified model name, and hyperparamter if appliable
# will return the calculated error
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
            rmse_linear = linear_reg_train(X1, y1, X2, y2)
            print(f'rmse={rmse_linear}')
            rmse.append(rmse_linear)

    # return average rmse
    return sum(rmse) / k


def gradient_descent(X, y, stepsize, gradient_function, starting_theta = None):
    n, d = X.shape
    theta = starting_theta
    if starting_theta is None:
        theta = np.ones((d, 1))  # right now starts from 1; possibly changed to random to avoid local minimum

    precision = 1
    max_iteration = 10000
    curr_iteration = 0
    while curr_iteration < max_iteration:
        grad = gradient_function(X, y, theta)
        step = stepsize * grad
        # print(step)
        theta = np.subtract(theta, step)
        if np.linalg.norm(step) < precision:
            print(f'converged after {curr_iteration} iterations')
            break

        if curr_iteration < 10 or curr_iteration % 100 == 0:
            print(f'iter={curr_iteration}, norm(step)={np.linalg.norm(step)}, norm(theta)={np.linalg.norm(theta)}')
            #             # print(theta)
        curr_iteration += 1
    if curr_iteration >= max_iteration:
        print(f'unable to converge after {curr_iteration} iterations')
    return theta


def linear_reg_gradient(X, y, theta):
    n, d = X.shape
    pred = np.matmul(X, theta)
    vector_y = np.reshape(y, (-1, 1))

    pred = np.maximum(pred, np.zeros((n, 1)))
    pred = np.subtract(pred, vector_y)
    return np.matmul(np.transpose(X), pred)


# train the linear regression model using X_train, y_train
# test on X_test, y_test, returns rmse
def linear_reg_train(X_train, y_train, X_test, y_test):
    n, d = X_train.shape

    # training
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    reg_theta = np.asarray(reg.coef_)
    reg_theta = np.reshape(reg_theta, (-1, 1))
    gradient = lambda a, b, t: linear_reg_gradient(a, b, t)
    theta = gradient_descent(X_train, y_train, 1 * (10**-6), gradient, reg_theta)
    # theta = reg_theta
    # theta = np.zeros((d, 1))

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
    # test_file_name = 'test_v2.csv'
    X, y = read_train_data(train_file_name, 24000) # read the first <num> rows of training data
    # X_test = read_test_data(test_file_name, 10000) # read the first 10000 rows of test data

    # experiment with different models / hyperparameters, observes rmse
    linear_rmse = train(X, y, model='linear')


