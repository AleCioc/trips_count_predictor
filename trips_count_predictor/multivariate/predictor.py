import pandas as pd

from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import percentage_error
from trips_count_predictor.multivariate.errors import r2_score


class TimeSeriesPredictor():

	def __init__(
			self,
			X_test,
			y_test,
			trainer_config,
			trainer,
	):

		self.X_test = X_test
		self.y_test = y_test
		self.trainer_config = trainer_config

		self.trainer = trainer
		self.start = trainer_config["start"]
		self.depth = trainer_config["depth"]

		self.X_test = self.X_test.loc[:, trainer.chosen_features].astype(float).dropna()
		self.y_test = self.y_test.astype(float).dropna()
		self.y_hat_test = pd.Series()
		self.results_dict = {}

	def predict(self):
		self.y_hat_test = pd.Series(
			self.trainer.pipeline.predict(self.X_test),
			index=self.X_test.index
		)

	def get_performance(self):
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
		return pd.Series(self.results_dict)
