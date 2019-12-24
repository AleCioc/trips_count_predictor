#Plotting functions
import os

import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.graphics.tsaplots as tsaplots
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from trips_count_predictor.config.config import root_figures_path

#Plot the autocorrelation function of the series of values for the vehicle
# def plot_autocorrelation(target):
#
# 	fig1, ax = plt.subplots(figsize=(16,9))
# 	lags=24
# 	#Set how many lags you want plotted
# 	tsaplots.plot_acf(target, ax, zero=True, lags=lags)
# 	plt.xlabel('Lags', fontsize = 16, labelpad=15)
# 	plt.ylabel('Partial autocorrelation value', fontsize = 16, labelpad=15)
# 	plt.title('Partial autocorrelation plot of Rentals', fontsize = 18, y=1.03)
# 	plt.xticks(range(0,lags+1), range(0,lags+1))
# 	plt.ylim([-1.2, 1.2])
# 	ax.tick_params(axis='y', labelsize=14)
# 	ax.tick_params(axis='x', labelsize=14)
# 	plt.grid()
# 	plt.tight_layout()



def plot_zscore(y_true, y_pred):
#    from scipy import stats
    residuals = y_true - y_pred
    z_score = (residuals - residuals.mean())/residuals.std()

    fig1, ax2 = plt.subplots(figsize=(13,8))
    plt.hist(z_score, bins=50)
    plt.grid()
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    plt.xlabel('Z-score', fontsize=16, labelpad=15)
    plt.ylabel('Frequency', fontsize=16, labelpad=15)
    plt.title('Z-scores of residual errors', fontsize = 18, y=1.03)
    plt.tight_layout()
    plt.grid()
#    plt.savefig()



#Residual errors plot in time
def plot_residuals(y_true, y_pred):

    residuals = y_true - y_pred
    residuals = pd.DataFrame(residuals)
    print('\nResidual errors description:')
    print(residuals.describe())

    fig1, ax1 = plt.subplots(figsize=(13,8))
    plt.title('Residual errors in time', fontsize = 18, y=1.03)
    plt.plot(residuals)
    plt.grid()
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    plt.xlabel('Model no.', fontsize=16, labelpad=15)
    plt.ylabel('Residual error', fontsize=16, labelpad=15)
    plt.tight_layout()
    plt.grid()
#    plt.savefig()



#Residual errors histogram plot
def plot_residuals_hist(y_true, y_pred):

    fig2, ax2 = plt.subplots(figsize=(9,6))
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50)
    plt.grid(axis='both')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    plt.xlabel('Residual error', fontsize=16, labelpad=15)
    plt.ylabel('Frequency', fontsize=16, labelpad=15)
    plt.title('Residual errors histogram', fontsize = 18, y=1.03)
    plt.tight_layout()

#    plt.savefig()


def plot_result(y_data, y_pred_series, alg):

    fig, ax = plt.subplots(figsize=(15,7))
    plt.title('Real values v.s predicted values', fontsize = 18, y=1.03)
    plt.ylabel('Number of trips', fontsize=16, labelpad=15)

    plt.xlabel('Hours', fontsize=16, labelpad=15)

    if alg == 'lr': label_alg = 'Linear Regression'
    elif alg == 'ridge': label_alg = 'Ridge Regression'
    elif alg == 'omp': label_alg = 'Orthogonal Matching Pursuit'
    elif alg == 'brr': label_alg = 'Bayesian Ridge Regression'
    elif alg == 'lsvr': label_alg = 'Linear Support Vector Regression'
    elif alg == 'svr': label_alg = 'Support Vector Regression'
    elif alg == 'rf': label_alg = 'Random Forest Regression'

    ax.plot(y_data, color='g', label='True Values')
    ax.plot(y_pred_series, color='coral', label=label_alg)

    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.legend(loc=1, fontsize=14)
    plt.tight_layout()
    plt.grid()


#Scatterplot of the real values v.s the predicted values
def plot_result_scatter(y_true, y_pred, alg):

    fig1, ax1 = plt.subplots(figsize=(7,7))
    plt.title('Real values v.s predicted values ', fontsize = 18, y=1.03)
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
    plt.grid()
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    plt.xlabel('Real y', fontsize=16, labelpad=15)
    plt.ylabel('Estimated y', fontsize=16, labelpad=15)
    plt.tight_layout()
    plt.grid()
#    plt.savefig()


#Stemplot of the coefficients of one model - if the algorithm has coeffiecients (SVR doesn't have any)
def plot_coefficients1(coef_series):

    fig, ax = plt.subplots(figsize=(13,7))
#    plt.stem(df_coef.iloc[line], use_line_collection=True)
    coef_series.plot.bar()
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.xlabel('Features', fontsize = 16, labelpad=15)
    plt.ylabel('Weight of feature', fontsize = 16)
    plt.xticks(range(0,len(coef_series)), coef_series.index, rotation=45)
    plt.title("Coefficients for the last model", fontsize=18, y=1.03)
    plt.tight_layout()
    plt.grid()
#    plt.savefig()


#Plot the coefficients of all models in time
def plot_coefficients2(df_coef, expanding=False):

    ax2 = plt.axes()
    df_coef.plot(figsize=(13,8), ax=ax2)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    plt.xlabel('Model number', fontsize = 16, labelpad=15)
    plt.ylabel('Coefficient magnitude', fontsize = 16)
#    plt.ylim((-0.7,2.6))

    if expanding: e = 'expanding'
    else: e = 'sliding'

    plt.legend(prop={'size': 14})
    plt.title("Change of coefficients | window: %s" % (e), fontsize=18, y=1.03)
    plt.tight_layout()
    plt.grid()
    plt.show()
#    plt.savefig()






