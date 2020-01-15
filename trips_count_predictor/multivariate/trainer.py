import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

from trips_count_predictor.multivariate.hyperparams_grids import hyperparams_grids
from trips_count_predictor.multivariate.best_hyperparams import best_hyperparams


class TimeSeriesTrainer:

	def __init__(
			self,
			X_train,
			y_train,
			trainer_config
	):

		self.X = X_train.dropna()
		self.y = y_train.dropna()

		self.config = trainer_config

		self.regr_type = trainer_config["regr_type"]
		self.regr = None

		self.scaler_type = trainer_config["scaler_type"]
		self.dim_red_type = trainer_config["dim_red_type"]
		self.dim_red_param = trainer_config["dim_red_param"]
		self.dim_reduction = None
		self.chosen_features = self.X.columns
		self.dim_red_scores = pd.DataFrame(index=self.X.columns)

		self.scaler = None
		self.scorers = None
		self.hyperparams_grid = None
		self.search = None

		self.get_regressor()
		self.steps = []
		self.pipeline = None
		self.best_regressor = None
		self.best_hyperparams = {}
		self.regression_coefs = pd.Series()
		self.cv_results = pd.DataFrame()

	def get_scaler(self):
		if self.scaler_type == "std":
			self.scaler = StandardScaler()
		if self.scaler_type == "minmax":
			self.scaler = MinMaxScaler()

	def get_regressor(self):
		if self.regr_type == "lr":
			self.regr = LinearRegression()
		elif self.regr_type == "ridge":
			self.regr = Ridge()
		elif self.regr_type == "omp":
			self.regr = OrthogonalMatchingPursuit()
		elif self.regr_type == "brr":
			self.regr = BayesianRidge()
		elif self.regr_type == "lsvr":
			self.regr = LinearSVR()
		elif self.regr_type == "svr":
			self.regr = SVR()
		elif self.regr_type == "rf":
			self.regr = RandomForestRegressor()

	def get_dim_reduction(self):

		if self.dim_red_param > len(self.X.columns):
			self.dim_red_param = len(self.X.columns)

		def crosscorr(X, y):
			return pd.DataFrame(X).apply(lambda x: x.corr(pd.Series(y))).values

		if self.dim_red_type == "crosscorr":
			self.dim_reduction = SelectKBest(
				crosscorr,
				self.dim_red_param
			)
		elif self.dim_red_type == "mutinf":
			self.dim_reduction = SelectKBest(
				mutual_info_regression,
				self.dim_red_param
			)
		elif self.dim_red_type == "pca":
			self.dim_reduction = PCA(
				n_components=self.dim_red_param
			)

	def get_scorers(self):
		self.scorers = {
			"mean_absolute_error": make_scorer(mean_absolute_error),
		}

	def get_hyperparams_grid(self):

		self.hyperparams_grid = hyperparams_grids[self.regr_type]

		new_grid = {}
		for k in self.hyperparams_grid.keys():
			new_grid["regressor__" + str(k)] = self.hyperparams_grid[k]
		self.hyperparams_grid = new_grid

	def get_hyperparams_grid_search(self):
		self.search = GridSearchCV(
			self.pipeline,
			self.hyperparams_grid,
			cv=3,
			return_train_score=False,
		)

	def get_feature_importances(self):

		if self.dim_red_type in ["crosscorr", "mutinf"]:
			self.chosen_features = self.X.loc[:, self.dim_reduction.get_support()].columns
			self.dim_red_scores = pd.Series(self.dim_reduction.scores_, index=self.X.columns)

		if self.regr_type in ['lr', 'lsvr']:
			self.regression_coefs = pd.Series(
				self.best_regressor.coef_,
				index=self.chosen_features
			)
		elif self.regr_type in ['rf']:
			self.regression_coefs = pd.Series(
				self.best_regressor.feature_importances_.tolist(),
				index=self.chosen_features
			)

	def get_best_params_regressor(self):

		if self.config["hyperparams_tuning"] != "best":
			best_hyperparams = {}
			for k in self.best_hyperparams.keys():
				best_hyperparams[k.split("__")[1]] = self.best_hyperparams[k]
		else:
			best_hyperparams = self.best_hyperparams

		if self.regr_type == "lr":
			return LinearRegression(**best_hyperparams)
		elif self.regr_type == "lsvr":
			return LinearSVR(**best_hyperparams)
		elif self.regr_type == "rf":
			return RandomForestRegressor(**best_hyperparams)

	def run(self):

		self.get_scorers()
		self.get_scaler()
		self.get_dim_reduction()


		if self.config["hyperparams_tuning"] == 1:
			if self.config["scaler_type"] != "":
				self.steps.append(("scaler", self.scaler))
			self.steps.append(("dim_reduction", self.dim_reduction))
			self.steps.append(("regressor", self.regr))
			self.pipeline = Pipeline(self.steps)
			self.get_hyperparams_grid()
			self.get_hyperparams_grid_search()
			self.search.fit(self.X, self.y)
			self.cv_results = self.search.cv_results_
			self.best_hyperparams = self.search.best_params_
			self.best_pipeline = self.search.best_estimator_
			self.best_regressor = self.best_pipeline.named_steps["regressor"]
			self.search.fit(self.X.loc[:, self.chosen_features], self.y)
			self.final_estimator = self.best_pipeline
		else:
			if self.config["hyperparams_tuning"] == "best" and self.regr_type in best_hyperparams:
				self.best_hyperparams = best_hyperparams[self.config["regr_type"]]
				self.best_regressor = self.get_best_params_regressor()
				self.steps = []
				if self.config["scaler_type"] != "":
					self.steps.append(("scaler", self.scaler))
				self.steps.append(("dim_reduction", self.dim_reduction))
				self.steps.append(("regressor", self.best_regressor))
				self.pipeline = Pipeline(self.steps)
				self.pipeline.fit(self.X, self.y)
			else:
				if self.config["scaler_type"] != "":
					self.steps.append(("scaler", self.scaler))
				self.steps.append(("dim_reduction", self.dim_reduction))
				self.steps.append(("regressor", self.regr))
				self.pipeline = Pipeline(self.steps)
				self.best_regressor = self.regr
				self.best_hyperparams = self.regr.get_params()
				self.pipeline.fit(self.X, self.y)
			self.final_estimator = self.pipeline
		self.get_feature_importances()
