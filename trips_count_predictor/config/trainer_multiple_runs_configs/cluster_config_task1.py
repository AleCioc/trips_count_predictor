multiple_runs_cluster_config = {
	"start": [1],
	"depth": [24],
	"training_policy": ["sliding", "expanding"],
	"training_size": [168*k for k in range(1, 6)],
	"training_update_t": [24],
	"regr_type": ["lr", "ridge", "rf", "svr"],
	"dim_red_type": [""],
	"dim_red_param": [0],
	"use_weather": [1],
	"use_y": [1],
	"use_calendar": [1]
}

