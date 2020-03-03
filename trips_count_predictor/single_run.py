import warnings
import os
import json
import sys

import pandas as pd

from trips_count_predictor.multivariate.model_validator import run_model_validator
from trips_count_predictor.city_loader.city_loader import CityLoader
from trips_count_predictor.config.config import trainer_single_run_configs_path


from city_data_manager.city_geo_trips.minneapolis_geo_trips import MinneapolisGeoTrips

grouped_trips_count = pd.DataFrame()
for month in range(5, 9):
	minneapolis = MinneapolisGeoTrips("city_of_minneapolis", 2019, month)
	minneapolis.load_resampled()
	grouped_trips_count = pd.concat([
		grouped_trips_count,
		minneapolis.resampled_origins
	])
trips_count_top_k_zones = pd.Series(
	grouped_trips_count.loc[
		:, grouped_trips_count.sum().sort_values().tail(1).index
	].sum(axis=1),
	name="count"
)
print(trips_count_top_k_zones)

# loader = CityLoader("Minneapolis")
# trips_count = pd.Series(name="count")
# for month in range(5, 6):
# 	trips_count = pd.concat([
# 		trips_count,
# 		loader.load_resampled_trips_data("city_of_minneapolis", 2019, month, '1h')
# 	])
# trips_count = trips_count.sort_index()
# print(trips_count)

config_path = os.path.join(
	trainer_single_run_configs_path,
	sys.argv[1]
)

with open(config_path, 'r') as f:
	trainer_single_run_config = json.load(f)

validators_input_dict = {
	"trips_count": trips_count_top_k_zones,
	"trainer_single_run_config": trainer_single_run_config
}

validator_summary = run_model_validator(validators_input_dict)
