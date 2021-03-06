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

	if trainer_config['use_calendar'] != 0:
		df_features["weekday"] = [int(date_column[i].strftime("%w")) for i in range(len(date_column))]
		df_features["hour"] = [int(date_column[i].strftime("%H")) for i in range(len(date_column))]
		df_features["weekend"] = df_features["weekday"]
		df_features["weekend"] = df_features["weekend"].replace([1, 2, 3, 4, 5], 1)
		df_features["weekend"] = df_features["weekend"].replace([0, 6], 2)
		df_features["weekend"] = df_features["weekend"]
		if type(trainer_config['use_calendar']) == list:
			df_features = df_features[trainer_config['use_calendar']]

	if trainer_config['use_weather'] != 0:
		loader = CityLoader("Minneapolis")
		weather = pd.DataFrame()
		for month in range(5, 9):
			weather = pd.concat([
				weather,
				loader.load_resampled_weather_data("asos", 2019, month, '1h')
			])
		if type(trainer_config['use_weather']) == list:
			weather = weather[trainer_config['use_weather']]
		df_features = df_features.join(weather)

	if trainer_config['use_y'] != 0:
		df_features = pd.concat([
			df_features, get_past_lags(trips_count, start, depth)],
			axis=1, sort=False
		)
		if type(trainer_config['use_y']) == list:
			df_features = df_features[trainer_config['use_y']]

	return df_features.dropna(axis=1, how="all").dropna()
