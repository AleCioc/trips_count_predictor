import os
import datetime

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import single_run_results_path
from trips_count_predictor.multivariate.trainer import TimeSeriesTrainer
from trips_count_predictor.multivariate.predictor import TimeSeriesPredictor
from trips_count_predictor.multivariate.data_prep import create_df_features
from trips_count_predictor.multivariate.data_prep import filter_df_features
from trips_count_predictor.multivariate.errors import print_errors
from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import percentage_error
from trips_count_predictor.multivariate.errors import r2_score
from trips_count_predictor.multivariate.plotter import TimeSeriesRegressionPlotter

def run_model_validator (validators_input_dict):

	trips_count = validators_input_dict["trips_count"]
	trainer_single_run_config = validators_input_dict["trainer_single_run_config"]

	#One-step validator
	validator = ModelValidator(
		trips_count,
		trainer_single_run_config
	)
	validator.run()
	return validator.get_output()


class ModelValidator:

	def __init__ (
		self,
		trips_count,
		trainer_config
	):

		self.trips_count = trips_count
		self.trainer_config = trainer_config

		self.start = trainer_config["start"]
		self.depth = trainer_config["depth"]

		self.use_y = trainer_config["use_y"]
		self.use_weather = trainer_config["use_weather"]
		self.use_calendar = trainer_config["use_calendar"]

		self.training_policy = trainer_config["training_policy"]
		self.training_size = trainer_config["training_size"]
		self.training_update_t = trainer_config["training_update_t"]

		self.dim_red_type = trainer_config["dim_red_type"]
		self.dim_red_param = trainer_config["dim_red_param"]

		self.y_test = pd.Series()
		self.y_hat_test = pd.Series()

		self.X = create_df_features(
			self.trips_count,
			self.trainer_config
		)
		self.y = self.trips_count.loc[self.X.index]

		self.X_train = pd.DataFrame()
		self.X_test = pd.DataFrame()

		self.cv_results = pd.DataFrame()
		self.df_coef = pd.DataFrame(columns=self.X.columns)
		self.last_predictor = None

		self.results_dict = {}
		self.chosen_features = []
		self.best_params = []

	def run (self):

		# Do a new model every len(x)/tot hours
		# Max_train_size is sliding window size
		if self.training_policy == "sliding":
			tscv = TimeSeriesSplit(
				n_splits = (len(self.X)-1)//self.training_update_t,
				max_train_size = self.training_size
			)
		elif self.training_policy == "expanding":
			tscv = TimeSeriesSplit(
				n_splits = (len(self.X)-1)//self.training_update_t
			)

		start_time = datetime.datetime.now()
		split_seq_n_index = []

		for train_index, test_index in tscv.split(self.X):

			if self.training_policy == "sliding":
				if len(train_index) < self.training_size:
					continue

			if self.dim_red_type == "autocorr":
				self.X_train = filter_df_features(
						self.y.iloc[train_index].copy(),
						self.X.iloc[train_index].copy(),
						self.dim_red_param
				)
				self.chosen_features.append(self.X_train.columns)

				self.X_test = self.X.iloc[test_index].copy()
				self.X_test = self.X_test.loc[:, self.X_train.columns]

			else:
				self.X_train = self.X.iloc[train_index].copy()
				self.X_test = self.X.iloc[test_index].copy()

			trainer = TimeSeriesTrainer(
				self.X_train,
				self.y.iloc[train_index],
				self.trainer_config
			)
			trainer.run()
			self.best_params += [trainer.best_params]

			predictor = TimeSeriesPredictor(
				self.X_test,
				self.y.iloc[test_index],
				self.trainer_config,
				trainer
			)
			predictor.predict()

			self.y_test = pd.concat([self.y_test, predictor.y_test])
			self.y_hat_test = pd.concat([self.y_hat_test, predictor.y_hat_test])
			self.y_hat_test.loc[self.y_hat_test < 0] = 0
			self.cv_results = pd.concat(
				[self.cv_results, pd.DataFrame(trainer.cv_results)],
				ignore_index=True, sort=False
			)
			self.df_coef = pd.concat(
				[self.df_coef, pd.DataFrame(trainer.coefs).T],
				ignore_index=True, sort=False
			)
			split_seq_n_index += [self.y.iloc[test_index].index[0]]

		print(
			self.trainer_config["regr_type"],
			"validator exec time:",
			(datetime.datetime.now()-start_time).total_seconds()
		)

		self.validation_time = (datetime.datetime.now()-start_time).total_seconds()

#        self.df_coef = pd.DataFrame(
#                 self.df_coef.values,
#                 index=split_seq_n_index,
#                 columns=self.df_coef.columns
#         )

		# self.last_coefs = trainer.coefs
		# self.last_predictor = predictor
		self.output = self.get_output()
		self.save_output()

		self.regression_plotter = TimeSeriesRegressionPlotter(
				self.trips_count,
				self.y_test,
				self.y_hat_test,
				self.trainer_config,
				self.df_coef
		)
		self.regression_plotter.plot_charts()

	def get_output(self):

		self.results_dict = self.trainer_config.copy()

		self.results_dict["mae"] = mean_absolute_error(
			self.y_test, self.y_hat_test
		)
		self.results_dict["rmse"] = rmse(
			self.y_test, self.y_hat_test
		)
		self.results_dict["rel"] = percentage_error(
			self.y_test, self.y_hat_test
		)
		self.results_dict["r2"] = r2_score(
			self.y_test, self.y_hat_test
		)
		self.results_dict["mean_fit_time"] = self.cv_results.mean_fit_time.mean()
		self.results_dict["validation_time"] = self.validation_time

		return pd.Series(self.results_dict)

	def save_output (self):
		model_conf_string = "_".join([str(v) for v in self.trainer_config.values()])
		check_create_path(single_run_results_path)
		self.output_path = os.path.join(
			single_run_results_path,
			model_conf_string
		)
		check_create_path(self.output_path)
		self.output.to_csv(
			os.path.join(
				self.output_path,
				"output_series.csv"
			), header=True
		)
		pd.DataFrame(self.best_params).to_csv(
			os.path.join(
				self.output_path,
				"best_hyperparams.csv"
			)
		)
		self.cv_results.to_csv(
			os.path.join(
				self.output_path,
				"cv_results.csv"
			)
		)
		self.df_coef.to_csv(
			os.path.join(
				self.output_path,
				"df_coef.csv"
			)
		)
