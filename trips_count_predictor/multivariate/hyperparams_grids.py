hyperparams_grids = {
	"lr": {
		"normalize": [True, False],
		"fit_intercept": [True, False]
	},
	"ridge": {
		"normalize": [True, False],
		"fit_intercept": [True, False],
		"alpha": [0.01, 0.1, 1]
	},
	"omp": {
		"normalize": [True, False],
		"fit_intercept": [True, False],
	},
	"brr": {
		"normalize": [True, False],
		"fit_intercept": [True, False],
	},
	"lsvr": {
		"C": [1, 10, 100, 1000]
	},
	"svr": {
		"kernel": ["rbf"],
		"gamma": ["scale"],
		"C": [1, 10, 100, 1000]
	},
	"rf": {
		"random_state": [1],
		"n_estimators": [60, 80, 100],
	}
}
