import os
import platform
import numpy as np


def getFileSep():
    if platform.system() == "Windows":
        return r"\\"
    else:
        return r"/"

def parse_boolean(string):
    return string == "True" or string == "true"

def save_array(directory, filename, array):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    sep = getFileSep()
    np.save(directory + sep + filename, array)
