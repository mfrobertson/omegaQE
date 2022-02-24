import sys
import tools
import numpy as np
import pandas as pd


def get_N0(key1, key2, key3, field):
    print(f"Getting {field} Noise + foreground from SO experiment with projection={key1}, sensitivity={key2} and method={key3}")
    N0_file = rf"../data/SO_noise/nlkk_v3_1_0_{key1}_{key2}_fsky0p4_{key3}_lT30-3000_lP30-5000.dat"
    dict = {"TT": (1,8), "TE": (2,9), "EE": (3,10), "TB": (4,11), "EB": (5,12), "Pol": (6,13), "MV": (7,14)}
    phi_index, curl_index = dict[field]
    df = pd.read_csv(N0_file, sep="\s+", header=None)
    return np.array(df[phi_index]), np.array(df[curl_index])

def parse_boolean(string):
    return string == "True" or string == "true"

def main(projection, sensitivity, iterative, typ):

    if projection == "all":
        key1 = "deproj0"
    if sensitivity == "base":
        key2 = "SENS1"
    elif sensitivity == "goal":
        key2 = "SENS2"
    if iterative:
        key3 = "it"
    else:
        key3 = "qe"

    N0 = get_N0(key1, key2, key3, typ)

    folder = '_N0'
    filename = f"N0_foreground_SO_{projection}_{sensitivity}_{key3}_{typ}.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, N0)


if __name__ == "__main__":
    # projection: all
    # sensitivity: base, goal
    # iterative: True, False
    # typ: TT, TE, EE, TB, EB, Pol, MV

    args = sys.argv[1:]
    if len(args) != 4:
        print("Must supply arguments: projection sensitivity iterative typ")
    projection = args[0]
    sensitivity = args[1]
    iterative = parse_boolean(args[2])
    typ = args[3]
    main(projection, sensitivity, iterative, typ)
