import pandas as pd
import numpy as np
import math
import sys
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# read the first nrows rows of training data
# to read the whole file set nrows = 0
# returns data and labels in type of array
def read_train_data(filename, nrows=0, fill='min'):
    X = pd.DataFrame()
    if nrows <= 0:
        for chunk in pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, chunksize=10000):
            X = pd.concat([X, chunk], axis=0, ignore_index=True)
    else:
        X = pd.read_csv(filename, sep=',', warn_bad_lines=False, error_bad_lines=False, low_memory=False, nrows=nrows)
    X = np.asarray(X.values, dtype=float)
    X = fill_nan(X, fill)
    # shuffling the data
    np.random.seed(514)
    np.random.shuffle(X)

    data = np.asarray(X[:, 1:-4], dtype=float)
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]
    # data = np.asarray(X[:, 1:5], dtype=float)  # just use figure 1-5 for basic testing for now
    # Normalize every column of data so that it ranges from 0 to 1
    data_normed = (data - data.min(0)) / data.ptp(0)
    labels = np.asarray(X[:, -1], dtype=float)
    labels = np.reshape(labels, (-1, 1))
    return data_normed, labels


# to fill in the NAN values
def fill_nan(data, fill='min'):
    if fill == 'min':
        use = np.nanmin(data, axis=0)
    elif fill == 'mean' or fill == 'avg':
        use = np.nanmean(data, axis=0)
    elif fill == 'med' or fill == 'median':
        use = np.nanmedian(data, axis=0)
    elif fill == 'max':
        use = np.nanmax(data, axis=0)
    else:
        print('warning: invalid NaN filling option')
        return data
    inds = np.where(np.isnan(data))
    data[inds] = np.take(use, inds[1])  # Fill in NaN with <use>
    return data


# train using X, y, specified model name, and hyperparamter if applicable
# will return the calculated error
def train(X, y, model='linear', hyperparamter=0.0, verbose=False, data_size=100000,
          step_size=-7, precision=-5, max_iter=500):
    training_size = data_size
    test_start = 80000  # this number is the start index of test set.
    X_test = X[test_start:, :]
    y_test = y[test_start:]

    X = X[0:training_size, :]
    y = y[0:training_size]
    # for training data X, y, applying 5-fold cross-validation
    k = 5
    n, d = X.shape
    train_rmse = []
    validation_rmse = []
    test_rmse = []
    # divide them into training set (X1, y1) and test set(X2, y2)
    for i in range(k):
        n = len(X)
        T_list = range(math.floor(n * i / k), math.floor(n * (i + 1) / k))
        S_list = np.setdiff1d(range(n), T_list)
        X1 = X[S_list, :]
        y1 = y[S_list]
        X2 = X[T_list, :]
        y2 = y[T_list]
        if verbose:
            print(f'======== {k}-fold: Iteration {i} ========')
        # calls the corresponding function that trains the model using X1, y1
        # and evaluate the performance on X2, y2 (validation error)
        # and evaluate the performance on X_test, y_test (test error)
        if model == 'naive':
            train_rmse_naive, validation_rmse_naive, test_rmse_naive\
                = naive_train(X1, y1, X2, y2, X_test, y_test, verbose=verbose)
            train_rmse.append(train_rmse_naive)
            validation_rmse.append(validation_rmse_naive)
            test_rmse.append(test_rmse_naive)
        if model == 'linear':
            train_rmse_linear, validation_rmse_linear, test_rmse_linear\
                = linear_reg_train(X1, y1, X2, y2, X_test, y_test, verbose=verbose,
                                   step_size=step_size, precision=precision, max_iter=max_iter)
            train_rmse.append(train_rmse_linear)
            validation_rmse.append(validation_rmse_linear)
            test_rmse.append(test_rmse_linear)
        elif model == 'ridge':
            train_rmse_ridge, validation_rmse_ridge, test_rmse_ridge\
                = ridge_reg_train(X1, y1, X2, y2, X_test, y_test, hyperparamter, verbose=verbose,
                                  step_size=step_size, precision=precision, max_iter=max_iter)
            train_rmse.append(train_rmse_ridge)
            validation_rmse.append(validation_rmse_ridge)
            test_rmse.append(test_rmse_ridge)
        elif model == 'lasso':
            train_rmse_lasso, validation_rmse_lasso, test_rmse_lasso \
                = lasso_reg_train(X1, y1, X2, y2, X_test, y_test, hyperparamter, verbose=verbose,
                                  step_size=step_size, precision=precision, max_iter=max_iter)
            train_rmse.append(train_rmse_lasso)
            validation_rmse.append(validation_rmse_lasso)
            test_rmse.append(test_rmse_lasso)
        elif model == 'svr':
            train_rmse_svr, validation_rmse_svr, test_rmse_svr \
                = svr_train(X1, y1, X2, y2, X_test, y_test, hyperparamter, verbose=verbose)
            train_rmse.append(train_rmse_svr)
            validation_rmse.append(validation_rmse_svr)
            test_rmse.append(test_rmse_svr)

    avg_train_rmse = sum(train_rmse) / k
    avg_validation_rmse = sum(validation_rmse) / k
    avg_test_rmse = sum(test_rmse) / k
    return avg_train_rmse, avg_validation_rmse, avg_test_rmse


"""
Naive model that predicts 0
"""
def naive_train(X_train, y_train, X_vali, y_vali, X_test, y_test, verbose):
    """Training error calculation"""
    n, d = X_train.shape
    y_pred = np.zeros((n, 1))
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    train_abs_error = np.subtract(y_pred, y_train)
    train_mse = np.square(train_abs_error)
    train_mse_value = np.sum(train_mse) / n
    train_rmse_value = math.sqrt(train_mse_value)

    """Validation error calculation"""
    n, d = X_vali.shape
    y_pred = np.zeros((n, 1))
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    vali_abs_error = np.subtract(y_pred, y_vali)
    vali_mse = np.square(vali_abs_error)
    vali_mse_value = np.sum(vali_mse) / n
    vali_rmse_value = math.sqrt(vali_mse_value)

    """Test error calculation"""
    n, d = X_test.shape
    y_pred = np.zeros((n, 1))
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    abs_error = np.subtract(y_pred, y_test)
    mse = np.square(abs_error)
    mse_value = np.sum(mse) / n
    rmse_value = math.sqrt(mse_value)

    if verbose:
        print()
        print(f'TRAINING: Root Mean Square Error = {train_rmse_value}')
        print(f'VALIDATION: Root Mean Square Error = {vali_rmse_value}')
        print(f'TEST: Root Mean Square Error = {rmse_value}')
        print()
    return train_rmse_value, vali_rmse_value, rmse_value


"""
This part is for SVR
"""
def svr_train(X_train, y_train, X_vali, y_vali, X_test, y_test, c_value, verbose):
    """TODO: Train the model using X_train, y_train, and c_value"""

    """Training error calculation"""
    n, d = X_train.shape
    y_pred = np.zeros((n, 1))  # TODO: replace the right side with your prediction on X_train
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    train_abs_error = np.subtract(y_pred, y_train)
    train_mse = np.square(train_abs_error)
    train_mse_value = np.sum(train_mse) / n
    train_rmse_value = math.sqrt(train_mse_value)

    """Validation error calculation"""
    n, d = X_vali.shape
    y_pred = np.zeros((n, 1))  # TODO: replace the right side with your prediction on X_vali
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    vali_abs_error = np.subtract(y_pred, y_vali)
    vali_mse = np.square(vali_abs_error)
    vali_mse_value = np.sum(vali_mse) / n
    vali_rmse_value = math.sqrt(vali_mse_value)

    """Test error calculation"""
    n, d = X_test.shape
    y_pred = np.zeros((n, 1))  # TODO: replace the right side with your prediction on X_test
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    abs_error = np.subtract(y_pred, y_test)
    mse = np.square(abs_error)
    mse_value = np.sum(mse) / n
    rmse_value = math.sqrt(mse_value)

    if verbose:
        print()
        print(f'TRAINING: Root Mean Square Error = {train_rmse_value}')
        print(f'VALIDATION: Root Mean Square Error = {vali_rmse_value}')
        print(f'TEST: Root Mean Square Error = {rmse_value}')
        print()
    return train_rmse_value, vali_rmse_value, rmse_value


"""
Below are implementation for 3 regression models
"""
def gradient_descent(X, y, stepsize, precision, gradient_function, value_function, starting_theta=None, hyperparameter=0, max_iter=500, verbose=False):
    n, d = X.shape
    theta = starting_theta
    if starting_theta is None:
        theta = np.ones((d, 1))  # right now starts from 1; possibly changed to random to avoid local minimum

    max_iteration = max_iter
    curr_iteration = 0
    step = 0
    while curr_iteration < max_iteration:
        grad = gradient_function(X, y, theta, hyperparameter)
        step = stepsize * grad
        # print(step)
        theta = np.subtract(theta, step)
        if curr_iteration > 0 and np.linalg.norm(step) < precision * stepsize:
            if verbose:
                print(f'converged after {curr_iteration} iterations')
            break

        if (curr_iteration < 5 or curr_iteration % 100 == 0) and verbose:
            value = value_function(X, y, theta, hyperparameter)
            print(f'iter={curr_iteration}, norm(step)={np.linalg.norm(step)}, value function={value}')
            #             # print(theta)
        if curr_iteration == 4 and verbose:
            print("......")
        curr_iteration += 1
    if curr_iteration >= max_iteration and verbose:
        value = value_function(X, y, theta, hyperparameter)
        print(f'unable to converge step into {precision * stepsize} after {curr_iteration} iterations, status:')
        print(f'norm(step): {np.linalg.norm(step)}, value function={value}')
    return theta


"""
This part is for lasso Regression
"""
def lasso_reg_gradient(X, y, theta, alpha):
    n, d = X.shape
    pred = np.matmul(X, theta)
    vector_y = np.reshape(y, (-1, 1))

    grad = np.maximum(pred, np.zeros((n, 1)))
    grad = np.subtract(grad, vector_y)
    grad = np.matmul(np.transpose(X), grad)
    grad = grad + np.sign(theta) * alpha / 2
    return grad


def lasso_reg_value(X, y, theta, alpha):
    n, d = X.shape
    value = np.minimum(np.maximum(X.dot(theta), np.zeros((n, 1))), np.ones((n, 1)) * 100)
    value = 0.5 * np.sum(np.square(np.subtract(value, y)))
    value = value + abs(0.5*alpha*np.sum(theta))
    return value


# train the linear regression model using X_train, y_train
# test on X_test, y_test, returns train rmse and test rmse
def lasso_reg_train(X_train, y_train, X_vali, y_vali, X_test, y_test, alpha, verbose=False, step_size=-7, precision=2, max_iter=500):
    n, d = X_train.shape

    # training
    # reg = LinearRegression(fit_intercept=False)
    reg = Lasso(alpha=alpha, fit_intercept=False)
    reg.fit(X_train, y_train)
    reg_theta = np.asarray(reg.coef_)
    reg_theta = np.reshape(reg_theta, (-1, 1))
    theta = gradient_descent(X_train, y_train, stepsize=10**step_size, precision=10**precision, max_iter=max_iter,
                             gradient_function=lasso_reg_gradient, value_function=lasso_reg_value,
                             starting_theta=reg_theta, hyperparameter=alpha, verbose=verbose)

    if verbose:
        zero_count_reg = d - np.count_nonzero(reg_theta)
        zero_standard = 10**-3  # change this number to change the definition of zero value
        zero_count = (np.abs(theta) < zero_standard).sum()
        print(f"\n<lasso regression model> zeros in theta: {zero_count_reg}/{d}")
        print(f"\n<gradient descent> zeros in theta: {zero_count}/{d}")

    """Training error calculation"""
    y_pred = X_train.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    train_abs_error = np.subtract(y_pred, y_train)
    train_mse = np.square(train_abs_error)
    train_mse_value = np.sum(train_mse) / n
    train_rmse_value = math.sqrt(train_mse_value)

    """Validation error calculation"""
    n, d = X_vali.shape
    y_pred = X_vali.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    vali_abs_error = np.subtract(y_pred, y_vali)
    vali_mse = np.square(vali_abs_error)
    vali_mse_value = np.sum(vali_mse) / n
    vali_rmse_value = math.sqrt(vali_mse_value)

    """Test error calculation"""
    n, d = X_test.shape
    y_pred = X_test.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    abs_error = np.subtract(y_pred, y_test)
    mse = np.square(abs_error)
    mse_value = np.sum(mse) / n
    rmse_value = math.sqrt(mse_value)

    if verbose:
        print()
        print(f'TRAINING: Root Mean Square Error = {train_rmse_value}')
        print(f'VALIDATION: Root Mean Square Error = {vali_rmse_value}')
        print(f'TEST: Root Mean Square Error = {rmse_value}')
        print()
    return train_rmse_value, vali_rmse_value, rmse_value


"""
This part is for ridge Regression
"""
def ridge_reg_gradient(X, y, theta, alpha):
    n, d = X.shape
    pred = np.matmul(X, theta)
    vector_y = np.reshape(y, (-1, 1))

    grad = np.maximum(pred, np.zeros((n, 1)))
    grad = np.subtract(grad, vector_y)
    grad = np.matmul(np.transpose(X), grad)
    grad = grad + alpha * theta
    return grad


def ridge_reg_value(X, y, theta, alpha):
    n, d = X.shape
    value = np.minimum(np.maximum(X.dot(theta), np.zeros((n, 1))), np.ones((n, 1)) * 100)
    value = 0.5 * np.sum(np.square(np.subtract(value, y)))
    value = value + 0.5*alpha*np.sum(np.square(theta))
    return value


# train the linear regression model using X_train, y_train
# test on X_test, y_test, returns train rmse and test rmse
def ridge_reg_train(X_train, y_train, X_vali, y_vali, X_test, y_test, alpha, verbose=False, step_size=-7, precision=2, max_iter=500):
    n, d = X_train.shape

    # training
    reg = Ridge(alpha=alpha, fit_intercept=False)
    # reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)
    reg_theta = np.asarray(reg.coef_)
    reg_theta = np.reshape(reg_theta, (-1, 1))
    theta = gradient_descent(X_train, y_train, stepsize=10**step_size, precision=10**precision, max_iter=max_iter,
                             gradient_function=ridge_reg_gradient, value_function=ridge_reg_value, starting_theta=reg_theta,
                             hyperparameter=alpha, verbose=verbose)

    """Training error calculation"""
    y_pred = X_train.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    train_abs_error = np.subtract(y_pred, y_train)
    train_mse = np.square(train_abs_error)
    train_mse_value = np.sum(train_mse) / n
    train_rmse_value = math.sqrt(train_mse_value)

    """Validation error calculation"""
    n, d = X_vali.shape
    y_pred = X_vali.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    vali_abs_error = np.subtract(y_pred, y_vali)
    vali_mse = np.square(vali_abs_error)
    vali_mse_value = np.sum(vali_mse) / n
    vali_rmse_value = math.sqrt(vali_mse_value)

    """Test error calculation"""
    n, d = X_test.shape
    y_pred = X_test.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    abs_error = np.subtract(y_pred, y_test)
    mse = np.square(abs_error)
    mse_value = np.sum(mse) / n
    rmse_value = math.sqrt(mse_value)

    if verbose:
        print()
        print(f'TRAINING: Root Mean Square Error = {train_rmse_value}')
        print(f'VALIDATION: Root Mean Square Error = {vali_rmse_value}')
        print(f'TEST: Root Mean Square Error = {rmse_value}')
        print()
    return train_rmse_value, vali_rmse_value, rmse_value


"""
This part is for linear regression with max(theta*x, 0)
"""
def linear_reg_gradient(X, y, theta, alpha=0):
    n, d = X.shape
    pred = np.matmul(X, theta)
    vector_y = np.reshape(y, (-1, 1))

    pred = np.maximum(pred, np.zeros((n, 1)))
    pred = np.minimum(pred, np.ones((n, 1)) * 100)
    pred = np.subtract(pred, vector_y)
    return np.matmul(np.transpose(X), pred)


def linear_reg_value(X, y, theta, alpha=0):
    n, d = X.shape
    value = np.minimum(np.maximum(X.dot(theta), np.zeros((n, 1))), np.ones((n, 1)) * 100)
    value = 0.5 * np.sum(np.square(np.subtract(value, y)))
    return value


# train the linear regression model using X_train, y_train
def linear_reg_train(X_train, y_train, X_vali, y_vali, X_test, y_test, verbose=False, step_size=-7, precision=2, max_iter=500):
    n, d = X_train.shape

    # training
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)
    reg_theta = np.asarray(reg.coef_)
    reg_theta = np.reshape(reg_theta, (-1, 1))
    theta = gradient_descent(X_train, y_train, stepsize=10**step_size, precision=10**precision, max_iter=max_iter,
                             gradient_function=linear_reg_gradient, value_function=linear_reg_value, starting_theta=reg_theta, verbose=verbose)

    """Training error calculation"""
    y_pred = X_train.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    train_abs_error = np.subtract(y_pred, y_train)
    train_mse = np.square(train_abs_error)
    train_mse_value = np.sum(train_mse) / n
    train_rmse_value = math.sqrt(train_mse_value)

    """Validation error calculation"""
    n, d = X_vali.shape
    y_pred = X_vali.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    vali_abs_error = np.subtract(y_pred, y_vali)
    vali_mse = np.square(vali_abs_error)
    vali_mse_value = np.sum(vali_mse) / n
    vali_rmse_value = math.sqrt(vali_mse_value)

    """Test error calculation"""
    n, d = X_test.shape
    y_pred = X_test.dot(theta)
    y_pred = np.maximum(y_pred, np.zeros((n, 1)))
    y_pred = np.minimum(y_pred, np.ones((n, 1)) * 100)
    abs_error = np.subtract(y_pred, y_test)
    mse = np.square(abs_error)
    mse_value = np.sum(mse) / n
    rmse_value = math.sqrt(mse_value)

    if verbose:
        print()
        print(f'TRAINING: Root Mean Square Error = {train_rmse_value}')
        print(f'VALIDATION: Root Mean Square Error = {vali_rmse_value}')
        print(f'TEST: Root Mean Square Error = {rmse_value}')
        print()
    return train_rmse_value, vali_rmse_value, rmse_value


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Arguments required. Run with -h or --help for usage.")
        sys.exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linear", help="run linear regression model", action="store_true")
    parser.add_argument("-r", "--ridge", type=float, help="run ridge regression model with hyperparameter")
    parser.add_argument("-a", "--lasso", type=float, help="run LASSO regression model with hyperparameter")
    parser.add_argument("-s", "--svr", type=float, help="run SVR model with SVR as C")
    parser.add_argument("--naive", help="run naive model which always predicts 0", action="store_true")
    parser.add_argument("-v", "--verbose", help="run with debug output", action="store_true")
    parser.add_argument("-f", "--file", help="data file path. Default is \'train_v2.csv\'")
    parser.add_argument("-n", "--fill", help="method to fill NaN value", choices=["mean", "min", "max", "median"])
    parser.add_argument("-d", "--data", help="number of data used for 5-fold validation", type=int)
    parser.add_argument("--step", help="log10 step size of gradient descent (step size will be set to 10^STEP)", type=int)
    parser.add_argument("--precision", help="log10 precision of gradient descent (precision will be set to 10^STEP)", type=int)
    parser.add_argument("-m", "--maxiter", help="max number of iterations in gradient descent", type=int)

    args = parser.parse_args()

    file_name = 'train_v2.csv'
    if args.file is not None:
        file_name = args.file

    fill = 'min'
    if args.fill is not None:
        fill = args.fill
    X, y = read_train_data(file_name, 0, fill=fill)  # read the first <num> rows of training data

    training_data_size = 80000
    if args.data is not None:
        training_data_size = args.data

    step = -7 if args.step is None else args.step
    precision = 2 if args.precision is None else args.precision
    max_iter = 500 if args.maxiter is None else args.maxiter

    if args.naive:
        train_err, valid_err, test_err = train(X, y, model='naive', verbose=args.verbose, data_size=training_data_size,
                                               step_size=step, precision=precision, max_iter=max_iter)
        print('=====Result=====')
        print(f'model=naive, fill={fill}, training_size={training_data_size}')
        print(f'training rmse:  \t{train_err}')
        print(f'validation rmse:\t{valid_err}')
        print(f'test rmse:      \t{test_err}')
        print('================\n')
    if args.linear:
        train_err, valid_err, test_err = train(X, y, model='linear', verbose=args.verbose, data_size=training_data_size,
                                               step_size=step, precision=precision, max_iter=max_iter)
        print('=====Result=====')
        print(f'model=linear, fill={fill}, training_size={training_data_size}')
        print(f'training rmse:  \t{train_err}')
        print(f'validation rmse:\t{valid_err}')
        print(f'test rmse:      \t{test_err}')
        print('================\n')
    if args.ridge is not None and args.ridge >= 0:
        train_err, valid_err, test_err = train(X, y, model='ridge', hyperparamter=args.ridge, verbose=args.verbose,
                                               data_size=training_data_size, step_size=step, precision=precision, max_iter=max_iter)
        print('=======Result=======')
        print(f'model=ridge, fill={fill}, lambda={args.ridge}, training_size={training_data_size}')
        print(f'training rmse:  \t{train_err}')
        print(f'validation rmse:\t{valid_err}')
        print(f'test rmse:      \t{test_err}')
        print('====================\n')
    if args.lasso is not None and args.lasso >= 0:
        train_err, valid_err, test_err = train(X, y, model='lasso', hyperparamter=args.lasso, verbose=args.verbose,
                                               data_size=training_data_size, step_size=step, precision=precision, max_iter=max_iter)
        print('=======Result=======')
        print(f'model=lasso, fill={fill}, lambda={args.lasso}, training_size={training_data_size}')
        print(f'training rmse:  \t{train_err}')
        print(f'validation rmse:\t{valid_err}')
        print(f'test rmse:      \t{test_err}')
        print('====================\n')

    if args.svr is not None:
        train_err, valid_err, test_err = train(X, y, model='svr', hyperparamter=args.svr, verbose=args.verbose,
                                               data_size=training_data_size)
        print('=======Result=======')
        print(f'model=svr, fill={fill}, C={args.svr}')
        print(f'training rmse:  \t{train_err}')
        print(f'validation rmse:\t{valid_err}')
        print(f'test rmse:      \t{test_err}')
        print('====================\n')


