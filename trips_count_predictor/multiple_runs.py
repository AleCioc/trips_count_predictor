import warnings
import os
import json
import sys
import multiprocessing as mp

import pandas as pd

from trips_count_predictor.multivariate.model_validator import run_model_validator
from trips_count_predictor.city_loader.city_loader import CityLoader
from trips_count_predictor.config.config import n_cores_remote
from trips_count_predictor.config.config import trainer_multiple_runs_configs_path
from trips_count_predictor.config.config import multiple_runs_results_path
from trips_count_predictor.config.config_grid import ConfigGrid


warnings.simplefilter(action='ignore')

loader = CityLoader("Minneapolis")
trips_count = pd.Series(name="count")
for month in range(5, 9):
	trips_count = pd.concat([
		trips_count,
		loader.load_resampled_trips_data("city_of_minneapolis", 2019, month, '1h')
	])
trips_count = trips_count.sort_index()

config_path = os.path.join(
	trainer_multiple_runs_configs_path,
	sys.argv[1]
)

with open(config_path, 'r') as f:
	trainer_multiple_runs_config = json.load(f)
config_grid = ConfigGrid(trainer_multiple_runs_config)

validators_input_dicts_tuples = []
for i in range(len(config_grid.conf_list)):
	validators_input_dicts_tuples.append({
		"trips_count": trips_count,
		"trainer_single_run_config": config_grid.conf_list[i]
	})

with mp.Pool(n_cores_remote) as pool:
	validators_output_list = pool.map(
		run_model_validator,
		validators_input_dicts_tuples
	)

output_path = os.path.join(
	multiple_runs_results_path,
	sys.argv[1].split(".")[0] + ".csv"
)
pd.DataFrame(validators_output_list).to_csv(output_path)
