import pandas as pd

from trips_count_predictor.city_loader.city_loader import CityLoader


def get_past_lags(series, start, depth):
	return pd.concat([
		pd.Series(series.shift(i), name=series.name + "_-" + str(i))
		for i in range(start, start + depth)
	], axis=1).dropna()


def create_df_features(trips_count, trainer_config):

	date_column = trips_count.index
	start = trainer_config['start']
	depth = trainer_config['depth']
	df_features = pd.DataFrame(index=date_column)

	if trainer_config['use_calendar']:
		df_features["weekday"] = [int(date_column[i].strftime("%w")) for i in range(len(date_column))]
		df_features["hour"] = [int(date_column[i].strftime("%H")) for i in range(len(date_column))]
		df_features["weekend"] = df_features["weekday"]
		df_features["weekend"] = df_features["weekend"].replace([1,2,3,4,5], 1)
		df_features["weekend"] = df_features["weekend"].replace([0, 6], 2)
		df_features["weekend"] = df_features["weekend"]

	if trainer_config['use_weather']:
		loader = CityLoader("Minneapolis")
		weather = loader.load_resampled_trips_data(
			"city_of_minneapolis", 2019, 5, '1h'
		)
		df_features = df_features.join(weather)

	if trainer_config['use_y']:
		df_features = pd.concat([
			df_features, get_past_lags(trips_count, start, depth)],
			axis=1, sort=False
		).dropna()

	return df_features.drop("count", axis=1)


def filter_df_features(trips_count, df_features, threshold):
	y = trips_count[df_features.index]
	autocorr = df_features.astype(float).apply(lambda x: x.corr(y))
	correlated_cols = autocorr[abs(autocorr) > threshold].index
	df_features_filtered = df_features[correlated_cols]
	return df_features_filtered
