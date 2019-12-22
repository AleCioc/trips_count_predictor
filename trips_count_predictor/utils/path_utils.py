import os


def check_create_path (path):
	if not os.path.exists(path):
		os.mkdir(path)