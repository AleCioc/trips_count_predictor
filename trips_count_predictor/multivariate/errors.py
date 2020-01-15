import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = y_true.values, y_pred.values
    return np.mean(np.abs((y_true - y_pred)) / y_true) * 100


def sym_mape(y_true, y_pred):
    return 100 * np.mean(2*np.abs(y_true - y_pred)/(y_true + y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))


def max_absolute_error(y_true, y_pred):
    return np.max(np.abs((y_true - y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))


def mean_relative_error(y_true, y_pred):
    a = np.sum(abs(y_pred-y_true))
    b = np.sum(abs(y_true))
    return (a/b)*100


def percentage_error2(y_true, y_pred):
    return (rmse(y_true, y_pred)/y_true.mean())*100


def percentage_error3(y_true, y_pred):
    a = np.sum(np.sqrt(abs(y_pred**2 - y_true**2)))
    b = np.sum(np.abs(y_true))
    return (a/b)*100


def r2_score(y_true, y_pred):
    u = ((y_true - y_pred) ** 2).sum()
    v = np.sum((y_true - y_true.mean()) ** 2)
    r2_score = 1 - u/v
    return r2_score


def z_score(y_true, y_pred):
    residuals_ = y_true - y_pred
    z_score_ = (residuals_ - residuals_.mean())/residuals_.std()
    return z_score_, z_score_.mean()


def residuals(y_true, y_pred):
    return y_true-y_pred


def print_errors(y_true, y_pred):

    print('MAE %.2f' % mean_absolute_error(y_true,y_pred))
    print('RMSE %.2f' % rmse(y_true,y_pred))
    print('Percentage Error %.2f' % mean_relative_error(y_true, y_pred))
    print('R2 score ', r2_score(y_true, y_pred))
    print('\n')
