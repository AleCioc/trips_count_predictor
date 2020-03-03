from city_data_manager.city_geo_trips.minneapolis_geo_trips import MinneapolisGeoTrips

minneapolis = MinneapolisGeoTrips("city_of_minneapolis", 2019, 5)

# minneapolis.get_trips_od_gdfs()
# print(minneapolis.trips_origins.dtypes)
# print(minneapolis.trips_destinations.dtypes)
#
# minneapolis.get_squared_grid()
# print(minneapolis.squared_grid.dtypes)
#
# minneapolis.map_zones_on_trips(minneapolis.squared_grid)
# minneapolis.squared_grid = minneapolis.map_trips_on_zones(minneapolis.squared_grid)

# minneapolis.save_points_data()
# minneapolis.save_squared_grid()


minneapolis.load()
minneapolis.get_resampled_points_count_by_zone()
minneapolis.save_resampled_points_data()
