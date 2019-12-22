import os

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

from sklearn.externals import joblib

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

#        self.freq = trainer_single_run_configs["freq"]
#        self.aggfunc_id = trainer_single_run_configs["aggfunc_id"]

		self.config = trainer_config

		self.regr_type = trainer_config["regr_type"]
		self.regr = None

		# self.scaler_type = trainer_single_run_configs["scaler_type"]
		# self.scaler = None

		self.dim_red_type = trainer_config["dim_red_type"]
		self.dim_red_param = trainer_config["dim_red_param"]
		self.dim_reduction = None

		self.scorers = None
		self.cv = None

		self.y_hat = pd.Series()

		self.get_regressor()
		self.perf = pd.DataFrame()
		self.steps = []
		self.pipeline = None

	def get_scaler(self):
		if self.scaler_type == "std":
			self.scaler = StandardScaler()

	def get_regressor(self):

		lr_params = {
			'normalize' : True
		}

		rf_params = {
			 'n_estimators' : 85,
			 'random_state' : 42
		}

		if self.regr_type == "lr":
			self.regr = linear_model.LinearRegression(**lr_params)
		elif self.regr_type == "rf":
			self.regr = RandomForestRegressor(**rf_params)

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

	def get_cv(self, n_splits=2, test_size=0.25):
		self.cv = ShuffleSplit(n_splits, test_size, random_state=1)

	def get_feature_importances(self):

		if self.regr_type == 'lr':
			self.coefs = pd.Series(self.regr.coef_, index=self.X.columns)
		elif self.regr_type == 'rf':
			self.coefs = pd.Series(self.regr.feature_importances_.tolist(), index=self.X.columns)


	def run(self):

		self.get_scorers()
		# self.get_scaler()
		self.get_dim_reduction()

		# self.steps.append(("scaler", self.scaler))
		self.steps.append(("dim_reduction", self.dim_reduction))
		self.steps.append(("regressor", self.regr))
		self.pipeline = Pipeline(self.steps)
		# self.get_cv()
		# self.cv_results = pd.DataFrame(cross_validate(
		# 	self.pipeline,
		# 	self.X, self.y,
		# 	scoring=self.scorers,
		# 	cv=self.cv,
		# 	return_train_score=False)
		# )

		self.pipeline.fit(self.X, self.y)
		self.y_hat = pd.Series(self.pipeline.predict(self.X), index=self.X.index)
		self.get_feature_importances()


	def save_final_estimator (self):
		model_conf_string = "_".join([str(v) for v in self.config.values()])
		out_pickle_filename = os.path.join(
			model_pickles_path,
			model_conf_string
		)
		joblib.dump(
			self.pipeline._final_estimator,
			filename=out_pickle_filename
		)
