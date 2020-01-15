import os
import datetime

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as ARIMA_statsmodels

from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import single_run_results_path
from trips_count_predictor.multivariate.errors import mean_absolute_error
from trips_count_predictor.multivariate.errors import rmse
from trips_count_predictor.multivariate.errors import mean_relative_error
from trips_count_predictor.multivariate.errors import r2_score


class ARIMA:

	def __init__(
			self,
			trips_count,
			start,
			depth,
			arima_d,
			arima_q
	):

		self.y_test = trips_count
		self.start = start
		self.depth = depth
		self.arima_d = arima_d
		self.arima_q = arima_q

		self.y_hat_test = pd.Series()
		self.summary = pd.Series()

		model_conf_string = "arima"
		check_create_path(single_run_results_path)
		self.output_path = os.path.join(
			single_run_results_path,
			model_conf_string
		)
		check_create_path(self.output_path)

		self.validation_time = datetime.timedelta()

	def run(self):

		start_time = datetime.datetime.now()
		self.regr = ARIMA_statsmodels(self.y_test.astype(float), order=(self.depth, self.arima_d, self.arima_q))
		self.regr_fit = self.regr.fit(disp=0, maxiter=400, method='css')
		if self.start == 1:
			self.y_hat_test = pd.Series(self.regr_fit.fittedvalues, index=self.y_test.index)
		elif self.start > 1:
			pass
		self.y_hat_test.loc[self.y_hat_test < 0] = 0
		self.validation_time = (datetime.datetime.now()-start_time).total_seconds()

	def get_summary(self):
		summary_dict = {
			"regr_type": "arima",
			"start": self.start,
			"depth": self.depth,
			"arima_d": self.arima_d,
			"arima_q": self.arima_q,
			"r2": r2_score(y_test_err, y_hat_test_err),
			"rmse": rmse(y_test_err, y_hat_test_err),
			"mae": mean_absolute_error(y_test_err, y_hat_test_err),
			"mxae": max_absolute_error(y_test_err, y_hat_test_err),
			"mre": mean_relative_error(y_test_err, y_hat_test_err),
			"mape": mean_absolute_percentage_error(y_test_err, y_hat_test_err),
			"smape": sym_mape(y_test_err, y_hat_test_err),
			"validation_time": self.validation_time
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
