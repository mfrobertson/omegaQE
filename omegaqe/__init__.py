import os
import omegaqe.tools as tools

sep = tools.getFileSep()
dir_path = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = dir_path + f"{sep}cache_planck{sep}"
DATA_DIR = dir_path + f"{sep}data_planck{sep}"
# DATA_DIR = dir_path + f"{sep}..{sep}DEMNUnii{sep}data{sep}"
# CACHE_DIR = dir_path + f"{sep}..{sep}DEMNUnii{sep}cache{sep}"
# RESULTS_DIR = dir_path + f"{sep}_results{sep}"
# CAMB_FILE = "DEMNUnii"
CAMB_FILE = "Planck"
