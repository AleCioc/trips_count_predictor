from trips_count_predictor.city_loader.city_loader import CityLoader

loader = CityLoader("Minneapolis")

minneapolis_resampled_trips_df = loader.load_resampled_trips_data(
	"city_of_minneapolis", 2019, 5, '1h'
)
print (minneapolis_resampled_trips_df.shape)

minneapolis_resampled_weather_df = loader.load_resampled_weather_data(
	"asos", 2019, 5, '1h'
)
print (minneapolis_resampled_weather_df.shape)
