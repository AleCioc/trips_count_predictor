from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import root_data_path
from trips_count_predictor.config.config import root_results_path
from trips_count_predictor.config.config import root_config_path
from trips_count_predictor.config.config import root_figures_path

from trips_count_predictor.config.config import single_run_results_path
from trips_count_predictor.config.config import multiple_runs_results_path

check_create_path(root_data_path)
check_create_path(root_results_path)
check_create_path(root_config_path)
check_create_path(root_figures_path)

check_create_path(single_run_results_path)
check_create_path(multiple_runs_results_path)
