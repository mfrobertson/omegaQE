import sys
import tools
import numpy as np
import pandas as pd


def get_N(deproj, N_file, offest):
    df = pd.read_csv(N_file, sep="\s+", header=None)
    col = int(deproj[-1]) + 1
    return np.concatenate((np.zeros(offest), np.array(df[col][12:], dtype='d')))

def save(N, typ):
    sep = tools.getFileSep()
    folder = '_N0' + sep + f'SO_{sensitivity}'+sep+'exp'
    filename = f"N_{typ}.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, N)

def main(sensitivity):

    projection = "all"

    if projection == "all":
        key1 = "deproj0"
    if sensitivity == "base":
        key2 = "baseline"
    elif sensitivity == "goal":
        key2 = "goal"


    N_file =  rf"../data/SO_noise/exp_noise/SO_LAT_Nell_T_atmv1_{key2}_fsky0p4_ILC_CMB.txt"
    N = get_N(key1, N_file, 40)
    N[10:40] = N[40]   # Modifying noise at Ls 10 -> 39, setting equal to N at L of 40
    save(N, "TT")
    N_file = rf"../data/SO_noise/exp_noise/SO_LAT_Nell_P_{key2}_fsky0p4_ILC_CMB_E.txt"
    N = get_N(key1, N_file, 10)
    save(N, "EE")
    N_file = rf"../data/SO_noise/exp_noise/SO_LAT_Nell_p_{key2}_fsky0p4_ILC_CMB_B.txt"
    N = get_N(key1, N_file, 10)
    save(N, "BB")


if __name__ == "__main__":
    # TODO: Be aware, N_TT is manually modified between Ls of 10 -> 39
    # projection: all
    # sensitivity: base, goal

    args = sys.argv[1:]
    if len(args) != 1:
        print("Must supply arguments: sensitivity")
    # projection = args[0]
    sensitivity = args[0]
    # iterative = tools.parse_boolean(args[2])
    # main(projection, sensitivity)
    main(sensitivity)
