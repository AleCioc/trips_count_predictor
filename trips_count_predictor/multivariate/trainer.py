import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

from trips_count_predictor.config.config import model_pickles_path


class TimeSeriesTrainer:

	def __init__(
			self,
			X_train,
			y_train,
			trainer_config
	):

		self.X = X_train
		self.y = y_train

		self.config = trainer_config

		self.regr_type = trainer_config["regr_type"]
		self.regr = None

		# self.scaler_type = trainer_config["scaler_type"]
		self.dim_red_type = trainer_config["dim_red_type"]
		self.dim_red_param = trainer_config["dim_red_param"]
		self.dim_reduction = None

		# self.scaler = None
		self.scorers = None
		self.hyperparams_grid = None
		self.search = None

		self.get_regressor()
		self.steps = []
		self.pipeline = None
		self.best_regressor = None
		self.best_params = {}
		self.coefs = pd.Series()
		self.cv_results = pd.DataFrame()

	def get_scaler(self):
		if self.scaler_type == "std":
			self.scaler = StandardScaler()

	def get_regressor(self):

		# lr_params = {
		# 	'normalize': True
		# }
		#
		# rf_params = {
		# 	'n_estimators': 85,
		# 	'random_state': 42
		# }

		if self.regr_type == "lr":
			self.regr = LinearRegression()
		elif self.regr_type == "ridge":
			self.regr = Ridge()
		elif self.regr_type == "svr":
			self.regr = SVR()
		elif self.regr_type == "rf":
			self.regr = RandomForestRegressor()

	def get_dim_reduction(self):
		if self.dim_red_type == "mutinf":
			self.dim_reduction = SelectKBest(
				mutual_info_regression,
				self.dim_red_param
			)
		elif self.dim_red_type == "pca":
			self.dim_reduction = PCA(
				n_components=int(self.dim_red_param * len(self.X.columns))
			)

	def get_scorers(self):
		self.scorers = {
			"mean_absolute_error": make_scorer(mean_absolute_error),
		}

	def get_hyperparams_grid(self):

		if self.regr_type == "lr":
			self.hyperparams_grid = {
				"normalize": [True, False],
				"fit_intercept": [True, False]
			}
		elif self.regr_type == "ridge":
			self.hyperparams_grid = {
				"normalize": [True, False],
				"fit_intercept": [True, False],
				"alpha": [0.001, 0.01, 0.1, 1, 10]
			}
		elif self.regr_type == "svr":
			self.hyperparams_grid = {
				"gamma": ["scale"],
				"kernel": ["linear", "poly", "rbf"],
				"C": [100]
			}
		elif self.regr_type == "rf":
			self.hyperparams_grid = {
				"random_state": [1],
				"n_estimators": [60, 80, 100],
			}

		new_grid = {}
		for k in self.hyperparams_grid.keys():
			new_grid["regressor__" + str(k)] = self.hyperparams_grid[k]
		self.hyperparams_grid = new_grid

	def get_hyperparams_grid_search(self):
		self.search = GridSearchCV(
			self.pipeline,
			self.hyperparams_grid,
			cv=2,
			scoring=self.scorers,
			return_train_score=False,
			refit="mean_absolute_error",
			n_jobs=-1
		)

	def get_feature_importances(self):

		if self.regr_type in ['lr', 'ridge'] or\
		(self.regr_type == "svr" and self.best_params["regressor__kernel"] == "linear"):
			self.coefs = pd.Series(
				self.pipeline.named_steps["regressor"].coef_,
				index=self.X.columns
			)
		elif self.regr_type in ['rf']:
			self.coefs = pd.Series(
				self.pipeline.named_steps["regressor"].feature_importances_.tolist(),
				index=self.X.columns
			)

	def get_best_params_regressor(self, best_params):

		new_best_params = {}
		for k in best_params.keys():
			new_best_params[k.split("__")[1]] = self.hyperparams_grid[k]
		best_params = new_best_params

		if self.regr_type == "lr":
			return LinearRegression(**best_params)
		elif self.regr_type == "rf":
			return RandomForestRegressor(**best_params)

	def run(self):

		self.get_scorers()
		# self.get_scaler()
		self.get_dim_reduction()

		# self.steps.append(("scaler", self.scaler))
		self.steps.append(("dim_reduction", self.dim_reduction))
		self.steps.append(("regressor", self.regr))
		self.pipeline = Pipeline(self.steps)

		self.get_hyperparams_grid()
		self.get_hyperparams_grid_search()
		self.search.fit(self.X, self.y)
		self.cv_results = self.search.cv_results_
		self.best_regressor = self.get_best_params_regressor(self.search.best_params_)
		self.pipeline.named_steps["regressor"] = self.best_regressor
		self.best_params = self.search.best_params_

		self.pipeline.fit(self.X, self.y)
		self.get_feature_importances()
