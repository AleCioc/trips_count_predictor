import os
import datetime

import pandas as pd

from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import single_run_results_path
from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import mean_relative_error
from trips_count_predictor.multivariate.errors import r2_score


class BaselineModel:

	def __init__(
			self,
			trips_count,
			start
	):

		self.y_test = trips_count
		self.start = start

		self.y_hat_test = pd.Series()
		self.summary = pd.Series()

		model_conf_string = "baseline"
		check_create_path(single_run_results_path)
		self.output_path = os.path.join(
			single_run_results_path,
			model_conf_string
		)
		check_create_path(self.output_path)

	def run(self):

		start_time = datetime.datetime.now()
		self.y_hat_test = self.y_test.shift(self.start)
		self.validation_time = (datetime.datetime.now()-start_time).total_seconds()

	def get_summary(self):
		summary_dict = {
			"regr_type": "baseline",
			"start": self.start,
			"mae": mean_absolute_error(self.y_test, self.y_hat_test),
			"rmse": rmse(self.y_test, self.y_hat_test),
			"rel": mean_relative_error(self.y_test, self.y_hat_test),
			"r2": r2_score(self.y_test, self.y_hat_test),
			"validation_time":self.validation_time
		}
		self.summary = pd.Series(summary_dict)
		return self.summary

	def save_output(self):

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
