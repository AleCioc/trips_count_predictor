import json

import pandas as pd

from trips_count_predictor.multivariate.model_validator import run_model_validator

from trips_count_predictor.city_loader.city_loader import CityLoader

from trips_count_predictor.config.config import trainer_single_run_default_config_path

loader = CityLoader("Minneapolis")
trips_count = pd.Series(name="count")
for month in range(5, 9):
	trips_count = pd.concat([
		trips_count,
		loader.load_resampled_trips_data("city_of_minneapolis", 2019, month, '1h')
	])

trips_count = trips_count.sort_index()

#Get arguments of training ex. depth of past window
with open(trainer_single_run_default_config_path, 'r') as f:
	trainer_single_run_config = json.load(f)

validators_input_dict = {
	"trips_count": trips_count,
	"trainer_single_run_config": trainer_single_run_config
}

validator_output = run_model_validator(validators_input_dict)
