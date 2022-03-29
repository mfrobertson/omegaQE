import sys
import tools
import numpy as np
import pandas as pd


def get_SEDs():
    SED_file = rf"../data/sed_bethermin_2013.dat"
    df = pd.read_csv(SED_file, sep="\s+", header=None)
    return np.array(df)

def main():

    SED = get_SEDs()

    folder = '_SED'
    filename = f"Bethermin_2013.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, SED)


if __name__ == "__main__":
    main()
