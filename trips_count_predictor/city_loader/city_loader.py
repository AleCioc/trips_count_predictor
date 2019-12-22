import os

import pandas as pd

from trips_count_predictor.config.config import data_paths_dict
from trips_count_predictor.city_loader.provider_loaders.minneapolis import MinneapolisLoader
from trips_count_predictor.city_loader.provider_loaders.asos import ASOSLoader


class CityLoader:

	def __init__(self, city):
		self.city = city

	def load_raw_trips_data(self, provider, year=None, month=None):
		if provider == "city_of_minneapolis":
			return MinneapolisLoader().load_raw_trips_data(year, month)

	def load_norm_trips_data(self, provider, year, month):
		data_path = os.path.join(
			data_paths_dict["norm_trips_data"],
			self.city,
			provider,
			"_".join([str(year), str(month)]) + ".csv"
		)
		return pd.read_csv(data_path, index_col=0, parse_dates=["start_time", "end_time"])

	def load_resampled_trips_data(self, provider, year, month, freq):
		data_path = os.path.join(
			data_paths_dict["resampled_trips_data"],
			self.city,
			provider,
			"_".join([str(year), str(month), freq]) + ".csv"
		)
		return pd.read_csv(data_path, index_col=0, parse_dates=[0])["count"]

	def load_raw_weather_data(self, provider):
		if provider == "asos":
			return ASOSLoader(self.city).load_raw_weather_data()

	def load_norm_weather_data(self, provider, year, month):
		data_path = os.path.join(
			data_paths_dict["norm_weather_data"],
			self.city,
			provider,
			"_".join([str(year), str(month)]) + ".csv"
		)
		return pd.read_csv(data_path, index_col=0, parse_dates=[0])

	def load_resampled_weather_data(self, provider, year, month, freq):
		data_path = os.path.join(
			data_paths_dict["resampled_weather_data"],
			self.city,
			provider,
			"_".join([str(year), str(month), freq]) + ".csv"
		)
		return pd.read_csv(data_path, index_col=0, parse_dates=[0])
