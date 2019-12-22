import os

import numpy as np
import pandas as pd

from trips_count_predictor.config.config import data_paths_dict


class ASOSLoader:

	def __init__(self, city):

		self.city = city
		self.provider = "asos"
		if self.city in ["Minneapolis"]:
			self.timezone = "America/Chicago"
			self.filename = "weather_minneapolis_hourly.csv"
		self.raw_weather_data_path = os.path.join(
			data_paths_dict["raw_weather_data"],
			self.city
		)

	def load_raw_weather_data (self):
		weather_data_path = os.path.join(
			self.raw_weather_data_path,
			self.provider,
			self.filename
		)
		weather = pd.read_csv(
			weather_data_path,
			parse_dates=['valid']
		).set_index("valid").sort_index()
		weather = weather.tz_localize(self.timezone)
		return weather
