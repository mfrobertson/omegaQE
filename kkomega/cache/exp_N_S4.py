import sys
import tools
import numpy as np
import pandas as pd


def get_N(N_file, col, offest):
    df = pd.read_csv(N_file, sep="\s+", header=None)
    return np.concatenate((np.zeros(offest), np.array(df[col][:], dtype='d')))

def save(N, typ):
    sep = tools.getFileSep()
    folder = '_N0' + sep + f'S4_base'+sep+'exp'
    filename = f"N_{typ}.npy"
    print(f"Saving {filename} in {folder}")
    T_cmb = 2.7255
    N = N*(1e-6/T_cmb)**2    # Converting from muK^2 -> unitless
    tools.save_array(folder, filename, N)

def main():

    projection = "all"

    if projection == "all":
        key1 = "deproj0"


    N_file =  rf"../data/S4_noise/exp_nooise/S4_190604d_2LAT_T_default_noisecurves_{key1}_SENS0_mask_16000_ell_TT_yy.txt"
    N = get_N(N_file, 1, 40)
    N[2:40] = N[40]   # Modifying noise at Ls 10 -> 39, setting equal to N at L of 40
    save(N, "TT")
    N_file = rf"../data/S4_noise/exp_nooise/S4_190604d_2LAT_pol_default_noisecurves_{key1}_SENS0_mask_16000_ell_EE_BB.txt"
    N = get_N(N_file, 1, 10)
    N[2:10] = N[10]   # Modifying noise at Ls 2 -> 10, setting equal to N at L of 10
    save(N, "EE")
    N = get_N(N_file, 2, 10)
    N[2:10] = N[10]   # Modifying noise at Ls 2 -> 10, setting equal to N at L of 10
    save(N, "BB")


if __name__ == "__main__":
    # TODO: Be aware, N_TT is manually modified between Ls of 2 -> 39 and N_P is manually modified between Ls of 2 -> 9
    # projection: all

    # args = sys.argv[1:]
    # if len(args) != 1:
    #     raise ValueError("Must supply arguments: sensitivity")
    # # projection = args[0]
    # # iterative = tools.parse_boolean(args[2])
    # # main(projection)
    main()
