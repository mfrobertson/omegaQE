import os
import omegaqe.tools as tools

sep = tools.getFileSep()
dir_path = os.path.dirname(os.path.realpath(__file__))
# CACHE_DIR = dir_path + f"{sep}cache{sep}"
# DATA_DIR = dir_path + f"{sep}data{sep}"
CACHE_DIR = dir_path + f"{sep}..{sep}fullsky_sims{sep}cache"
DATA_DIR = dir_path + f"{sep}..{sep}fullsky_sims{sep}data"
# RESULTS_DIR = dir_path + f"{sep}results{sep}"
RESULTS_DIR = CACHE_DIR
LOGGING_DIR = dir_path + f"{sep}..{sep}fullsky_sims{sep}_output"
CAMB_FILE = "DEMNUnii"
