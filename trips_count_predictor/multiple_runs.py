import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import multiprocessing as mp

import pandas as pd
from sklearn.utils import parallel_backend

from trips_count_predictor.multivariate.model_validator import run_model_validator
from trips_count_predictor.city_loader.city_loader import CityLoader
from trips_count_predictor.config.config import n_cores_remote
from trips_count_predictor.config.config import default_results_path
from trips_count_predictor.config.config import cluster_results_path
from trips_count_predictor.config.config_grid import ConfigGrid
from trips_count_predictor.config.trainer_multiple_runs_configs.default_config import multiple_runs_default_config
from trips_count_predictor.config.trainer_multiple_runs_configs.cluster_config_task1 import multiple_runs_cluster_config

loader = CityLoader("Minneapolis")
trips_count = pd.Series(name="count")
for month in range(5, 9):
	trips_count = pd.concat([
		trips_count,
		loader.load_resampled_trips_data("city_of_minneapolis", 2019, month, '1h')
	])

if len(sys.argv) > 1:

	if sys.argv[1] == "cluster":
		config_grid = ConfigGrid(multiple_runs_cluster_config)
	else:
		config_grid = ConfigGrid(multiple_runs_default_config)

validators_input_dicts_tuples = []
for i in range(len(config_grid.conf_list)):
	validators_input_dicts_tuples.append({
		"trips_count": trips_count,
		"trainer_single_run_config": config_grid.conf_list[i]
	})

with parallel_backend('multiprocessing'):
	with mp.Pool(n_cores_remote) as pool:
		validators_output_list = pool.map(
			run_model_validator,
			validators_input_dicts_tuples
		)

if len(sys.argv) == 2:
	if sys.argv[1] == "cluster":
		pd.DataFrame(validators_output_list).to_csv(cluster_results_path)
else:
	pd.DataFrame(validators_output_list).to_csv(default_results_path)

