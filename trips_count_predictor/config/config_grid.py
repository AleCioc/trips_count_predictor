import itertools
import datetime
import numpy as np


class ConfigGrid():

	def __init__(self, json_conf_grid):

		self.conf_keys = json_conf_grid.values()
		self.conf_list = []
		for el in itertools.product(*json_conf_grid.values()):
			conf = {k: None for k in json_conf_grid}
			i = 0
			for k in conf.keys():
				conf[k] = el[i]
				i += 1
			self.conf_list += [conf]
