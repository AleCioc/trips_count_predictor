import pandas as pd

from trips_count_predictor.multivariate.errors import r2_score
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import max_absolute_error
from trips_count_predictor.multivariate.errors import mean_relative_error
from trips_count_predictor.multivariate.errors import mean_absolute_percentage_error
from trips_count_predictor.multivariate.errors import sym_mape


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

		self.summary = pd.Series()

	def predict(self):
		self.y_hat_test = pd.Series(
			self.trainer.final_estimator.predict(self.X_test[self.trainer.chosen_features]),
			index=self.X_test.index
		)

	def get_performance(self):
		y_test_err = self.y_test.loc[self.y_test > 0].iloc[24:-24]
		y_hat_test_err = self.y_hat_test.loc[y_test_err.index]
		errs = y_test_err - y_hat_test_err
		summary_dict = {
			"r2": r2_score(y_test_err, y_hat_test_err),
			"rmse": rmse(y_test_err, y_hat_test_err),
			"mae": mean_absolute_error(y_test_err, y_hat_test_err),
			"mxae": max_absolute_error(y_test_err, y_hat_test_err),
			"mre": mean_relative_error(y_test_err, y_hat_test_err),
			"mape": mean_absolute_percentage_error(y_test_err, y_hat_test_err),
			"smape": sym_mape(y_test_err, y_hat_test_err),
		}
		self.summary = pd.Series(summary_dict)
		return self.summary
