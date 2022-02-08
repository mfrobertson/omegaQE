import lensit as li
import numpy as np
import os
import platform
import sys


def lensit_cache_dir_setup():
    if not 'LENSIT' in os.environ.keys():
        os.environ['LENSIT'] = '_tmp'


def getFileSep():
    if platform.system() == "Windows":
        return r"\\"
    else:
        return r"/"


def get_N0(exp, LDres, HDres, estimator):
    isocov = li.get_isocov(exp, LDres, HDres)
    return isocov.get_N0cls('TQU', isocov.lib_skyalm)


def save_array(directory, filename, array):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    sep = getFileSep()
    np.save(directory + sep + filename, array)

def main(exp, LDres, HDres, estimator):
    lensit_cache_dir_setup()
    N0 = get_N0(exp, LDres, HDres, estimator)

    folder = '_cache'
    filename = f"N0_{LDres}_{HDres}.npy"
    save_array(folder, filename, N0)


if __name__ == "__main__":
    # exp = experiment = {'SO', 'S4', etc}
    # LDres = is the number of pixels of a side = 2^LDres
    # LDres = 13 gives N = 8192
    # HDres = physical size in arcmins = 0.74*2^HDres
    # HDres = 14 ~= 12000 arcmin ~= 200deg ~= 48000Mpc
    # est = QE estimator = {'T', 'QU', 'TQU'}

    args = sys.argv[1:]
    if len(args) != 4:
        print("Must supply arguments: exp LDres HDres est")
    exp = args[0]
    LDres = int(args[1])
    HDres = int(args[2])
    est = args[3]
    main(exp, LDres, HDres, est)
