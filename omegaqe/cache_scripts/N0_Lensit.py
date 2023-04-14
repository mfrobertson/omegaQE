import lensit as li
import os
import sys
import omegaqe.tools as tools

cache_dir = omegaqe.CACHE_DIR

def lensit_cache_dir_setup():
    if not 'LENSIT' in os.environ.keys():
        os.environ['LENSIT'] = '_tmp'


def get_N0(exp, LDres, HDres, estimator):
    print(f"Getting Noise from experiment {exp} at resolution LDres={LDres}, HDres={HDres} for QE={estimator}")
    isocov = li.get_isocov(exp, LDres, HDres)
    return isocov.get_N0cls(estimator, isocov.lib_skyalm)


def main(exp, LDres, HDres, estimator):
    lensit_cache_dir_setup()
    N0 = get_N0(exp, LDres, HDres, estimator)

    folder = cache_dir + '_N0'
    filename = f"N0_{exp}_{LDres}_{HDres}_{estimator}.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, N0)


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
