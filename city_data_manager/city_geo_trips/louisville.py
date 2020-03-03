import os

import pandas as pd
import geopandas as gpd

from city_data_manager.city_data_source.trips_data_source.louisville_scooter_trips import LouisvilleScooterTrips
from city_data_manager.city_geo_trips.city_geo_trips import CityGeoTrips


class Louisville(CityGeoTrips):

	def __init__(self):

		super().__init__("Louisville")

		self.trips_ds = LouisvilleScooterTrips()
		self.trips_ds.load_raw()
		self.trips_ds.normalise()


louisville = Louisville()
print(louisville.trips_ds.trips_df_norm.columns)
