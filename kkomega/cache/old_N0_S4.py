import sys
import tools
import numpy as np
import pandas as pd


def get_N0(key1, field):
    print(f"Getting {field} Noise + foreground from S4 experiment with projection={key1}")
    N0_file = rf"../data/S4_noise/kappa_{key1}_sens0_16000_lT300-3000_lP300-5000.dat"
    dict = {"TT": (1,8), "TE": (2,9), "EE": (3,10), "TB": (4,11), "EB": (5,12), "Pol": (6,13), "MV": (7,14)}
    phi_index, curl_index = dict[field]
    df = pd.read_csv(N0_file, sep="\s+", header=None)
    return np.array(df[phi_index]), np.array(df[curl_index])

def main(projection, field):

    if projection == "all":
        key1 = "deproj0"

    N0 = get_N0(key1, field)

    folder = '_N0'
    filename = f"N0_foreground_S4_{projection}_{field}.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, N0)


if __name__ == "__main__":
    # projection: all
    # field: TT, TE, EE, TB, EB, Pol, MV

    args = sys.argv[1:]
    if len(args) != 2:
        print("Must supply arguments: projection field(s)")
    projection = args[0]
    field = args[1]
    main(projection, field)
