import os
import omegaqe.tools as tools

sep = tools.getFileSep()
dir_path = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = dir_path + f"{sep}cache{sep}"
DATA_DIR = dir_path + f"{sep}data{sep}"
RESULTS_DIR = dir_path + f"{sep}results{sep}"
CAMB_FILE = "Lensit"
