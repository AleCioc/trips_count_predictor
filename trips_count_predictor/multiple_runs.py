import pandas as pd
import multiprocessing as mp

from trips_count_predictor.multivariate.model_validator import run_model_validator

from trips_count_predictor.city_loader.city_loader import CityLoader

from trips_count_predictor.config.config import n_cores_remote
from trips_count_predictor.config.config import default_results_path
from trips_count_predictor.config.config_grid import ConfigGrid
from trips_count_predictor.config.trainer_multiple_runs_configs.default_config import multiple_runs_default_config


loader = CityLoader("Minneapolis")
trips_count = loader.load_resampled_trips_data(
	"city_of_minneapolis", 2019, 5, '1h'
)

config_grid = ConfigGrid(multiple_runs_default_config)
validators_input_dicts_tuples = []
for i in range(len(config_grid.conf_list)):
	print(config_grid.conf_list[i])
	validators_input_dicts_tuples.append({
		"trips_count": trips_count,
		"trainer_single_run_config": config_grid.conf_list[i]
	})

with mp.Pool(n_cores_remote) as pool:

	validators_output_list = pool.map(
		run_model_validator,
		validators_input_dicts_tuples
	)

pd.DataFrame(validators_output_list).to_csv(default_results_path)
