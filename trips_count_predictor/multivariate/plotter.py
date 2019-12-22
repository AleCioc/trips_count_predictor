import os

import pandas as pd
import matplotlib.pyplot as plt

from trips_count_predictor.utils.path_utils import check_create_path
from trips_count_predictor.config.config import root_figures_path
from trips_count_predictor.multivariate.plot_stuff import plot_result, plot_residuals_hist


class TimeSeriesRegressionPlotter():

    def __init__(
            self,
            trips_h,
            y_true,
            y_hat,
            trainer_config,
            df_coef
    ):

        self.trips_h = trips_h
        self.y_true = y_true
        self.y_hat = y_hat
        self.trainer_config = trainer_config
        self.df_coef = df_coef

        check_create_path(root_figures_path)

    def plot_charts(self):

        model_config_string = "_".join([str(v) for v in self.trainer_config.values()])

        plot_result(self.trips_h, self.y_hat, self.trainer_config["regr_type"])
        plt.tight_layout()
        plt.savefig(os.path.join(
            root_figures_path,
            model_config_string + "_result.png"
            )
        )
        plt.close()

        plot_residuals_hist(self.y_true, self.y_hat)
        plt.savefig(os.path.join(
            root_figures_path,
            model_config_string + "_residuals_hist.png"
            )
        )
        plt.close()

        # self.df_coef.plot.barh(stacked=True, figsize=(15,7))
        # plt.tight_layout()
        # plt.savefig(os.path.join(
        #     root_figures_path,
        #     model_config_string + "_all_coefs.png"
        #     )
        # )
        # plt.close()
		#
        # self.df_coef.mean().plot.bar(stacked=False, figsize=(15,7))
        # plt.tight_layout()
        # plt.savefig(os.path.join(
        #     root_figures_path,
        #     model_config_string + "_mean_coefs.png"
        #     )
        # )
        # plt.close()
