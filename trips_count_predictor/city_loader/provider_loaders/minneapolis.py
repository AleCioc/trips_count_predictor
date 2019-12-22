import os

import numpy as np
import pandas as pd

from trips_count_predictor.config.config import data_paths_dict


class MinneapolisLoader:

	def __init__(self):

		self.city = "Minneapolis"
		self.provider = "city_of_minneapolis"
		self.months_dict = {
			5: "May",
			6: "June",
			7: "July",
			8: "August",
		}
		self.raw_trips_data_path = os.path.join(
			data_paths_dict["raw_trips_data"],
			self.city
		)
		self.raw_weather_data_path = os.path.join(
			data_paths_dict["raw_weather_data"],
			self.city
		)

	def load_raw_trips_data(self, year, month):
		month = self.months_dict[month]
		year = str(year)
		data_path = os.path.join(
			self.raw_trips_data_path,
			self.provider,
			"_".join(["Motorized", "Foot", "Scooter", "Trips", month, year]) + ".csv"
		)
		df = pd.read_csv(data_path, parse_dates=[3, 4])
		df.StartTime = df.StartTime.dt.tz_convert('America/Chicago')
		df.EndTime = df.EndTime.dt.tz_convert('America/Chicago')
		return df
