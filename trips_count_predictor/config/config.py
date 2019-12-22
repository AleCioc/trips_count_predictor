import os

n_cores_remote = 40

root_data_path = os.path.join(
	os.path.dirname(os.path.dirname(__file__)),
	"data"
)

data_paths_dict = {

	"raw_trips_data": os.path.join(root_data_path, "raw_trips_data"),
	"raw_weather_data": os.path.join(root_data_path, "raw_weather_data"),

	"norm_trips_data": os.path.join(root_data_path, "norm_trips_data"),
	"norm_weather_data": os.path.join(root_data_path, "norm_weather_data"),

	"resampled_trips_data": os.path.join(root_data_path, "resampled_trips_data"),
	"resampled_weather_data": os.path.join(root_data_path, "resampled_weather_data"),

}

root_config_path = os.path.join(
	os.path.dirname(os.path.dirname(__file__)),
	"config"
)

trainer_single_run_configs_path = os.path.join(
	root_config_path,
	"trainer_single_run_configs"
)

trainer_single_run_default_config_path = os.path.join(
	trainer_single_run_configs_path,
	"default_config.json"
)

model_pickles_path = os.path.join(
	root_data_path,
	"model_pickles"
)

root_figures_path = os.path.join(
	os.path.dirname(os.path.dirname(__file__)),
	"figures"
)

root_results_path = os.path.join(
	os.path.dirname(os.path.dirname(__file__)),
	"results"
)

default_results_path = os.path.join(
		root_results_path,
		"multiple_runs_default_config_results.csv"
	)
