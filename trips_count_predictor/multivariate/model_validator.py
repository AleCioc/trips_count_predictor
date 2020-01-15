import os
import datetime

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import single_run_results_path
from trips_count_predictor.multivariate.trainer import TimeSeriesTrainer
from trips_count_predictor.multivariate.predictor import TimeSeriesPredictor
from trips_count_predictor.multivariate.data_prep import create_df_features
from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import percentage_error
from trips_count_predictor.multivariate.errors import r2_score
from trips_count_predictor.multivariate.plotter import TimeSeriesRegressionPlotter


def run_model_validator (validators_input_dict):

	trips_count = validators_input_dict["trips_count"]
	trainer_single_run_config = validators_input_dict["trainer_single_run_config"]

	validator = ModelValidator(
		trips_count,
		trainer_single_run_config
	)
	validator.run()
	return validator.get_summary()


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

		self.X = create_df_features(
			self.trips_count,
			self.trainer_config
		)
		self.X_train = pd.DataFrame()
		self.X_test = pd.DataFrame()

		self.y = self.trips_count.loc[self.X.index]
		self.y_test = pd.Series()
		self.y_hat_test = pd.Series()

		self.dim_red_type = trainer_config["dim_red_type"]
		self.dim_red_param = trainer_config["dim_red_param"]
		self.dim_red_scores = pd.DataFrame()
		self.feature_coefs_model = pd.DataFrame(columns=self.X.columns)
		self.chosen_features = []
		self.dim_red_scores = pd.DataFrame()

		self.cv_results = pd.DataFrame()
		self.best_hyperparams = []

		self.summary = pd.Series()

		model_conf_string = "_".join([str(v) for v in self.trainer_config.values()])
		check_create_path(single_run_results_path)
		self.output_path = os.path.join(
			single_run_results_path,
			model_conf_string
		)
		check_create_path(self.output_path)

		self.validation_time = datetime.timedelta()

	def run (self):

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

			if len(train_index) < self.training_size:
				continue

			X_train = self.X.iloc[train_index].copy()
			X_test = self.X.iloc[test_index].copy()

			trainer = TimeSeriesTrainer(
				X_train,
				self.y.iloc[train_index],
				self.trainer_config
			)
			trainer.run()
			self.best_hyperparams += [trainer.best_hyperparams]

			predictor = TimeSeriesPredictor(
				X_test,
				self.y.iloc[test_index],
				self.trainer_config,
				trainer
			)
			predictor.predict()

			self.X_test = pd.concat([self.X_test, predictor.X_test])
			self.y_test = pd.concat([self.y_test, predictor.y_test])
			self.y_hat_test = pd.concat([self.y_hat_test, predictor.y_hat_test])
			self.y_hat_test.loc[self.y_hat_test < 0] = 0
			self.cv_results = pd.concat(
				[self.cv_results, pd.DataFrame(trainer.cv_results)],
				ignore_index=True, sort=False
			)
			self.dim_red_scores = pd.concat(
				[self.dim_red_scores, pd.DataFrame(trainer.dim_red_scores).T],
				sort=False
			)
			self.feature_coefs_model = pd.concat(
				[self.feature_coefs_model, pd.DataFrame(trainer.regression_coefs).T],
				ignore_index=True, sort=False
			)
			split_seq_n_index += [self.y.iloc[test_index].index[0]]

		print(
			self.trainer_config["regr_type"],
			self.X_test.shape,
			(datetime.datetime.now()-start_time).total_seconds()
		)

		self.validation_time = (datetime.datetime.now()-start_time).total_seconds()
		self.summary = self.get_summary()
		self.best_hyperparams = pd.DataFrame(self.best_hyperparams)
		self.save_output()

		self.regression_plotter = TimeSeriesRegressionPlotter(
				self.trips_count,
				self.y_test,
				self.y_hat_test,
				self.trainer_config,
				self.feature_coefs_model
		)
		self.regression_plotter.plot_charts()

	def get_summary(self):

		summary_dict = self.trainer_config.copy()
		summary_dict.update({
			"mae": mean_absolute_error(self.y_test, self.y_hat_test),
			"rmse": rmse(self.y_test, self.y_hat_test),
			"rel": percentage_error(self.y_test, self.y_hat_test),
			"r2": r2_score(self.y_test, self.y_hat_test)
		})
		self.summary = pd.Series(summary_dict)
		if self.trainer_config["hyperparams_tuning"] == 1:
			summary_dict["mean_fit_time"] = self.cv_results.mean_fit_time.mean()
		summary_dict["validation_time"] = self.validation_time
		return pd.Series(summary_dict)

	def save_output (self):
		
		def _to_csv(pd_obj, csv_name):
			pd_obj.to_csv(
				os.path.join(
					self.output_path,
					csv_name
				), header=True
			)

		_to_csv(self.y_test, "y_test.csv")
		_to_csv(self.y_hat_test, "y_hat_test.csv")
		_to_csv(self.summary, "summary.csv")
		_to_csv(self.cv_results, "cv_results.csv")
		_to_csv(self.best_hyperparams, "best_hyperparams.csv")
		_to_csv(self.best_hyperparams.mode(), "best_hyperparams_mode.csv")
		_to_csv(self.dim_red_scores, "dim_red_scores.csv")
		_to_csv(self.dim_red_scores.mean(), "dim_red_scores_mean.csv")
		_to_csv(self.feature_coefs_model, "feature_coefs.csv")
		_to_csv(self.feature_coefs_model.mean(), "feature_coefs_mean.csv")
