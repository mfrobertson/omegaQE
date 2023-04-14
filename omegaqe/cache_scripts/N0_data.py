import pandas as pd
import numpy as np
import os

import omegaqe
import omegaqe.tools as tools


cache_dir = omegaqe.CACHE_DIR
data_dir = omegaqe.DATA_DIR
sep = tools.getFileSep()


def save_N0(exps, powerspectra, single_fields, gmv_fields, L_cuts, convert=False):
    columns = np.array(single_fields + gmv_fields)
    Ls = np.arange(2, 5001)
    for exp in exps:
        for ps in powerspectra:
            df_phi = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
            df_curl = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
            for iii, col in enumerate(columns):
                typ = "single" if iii < np.size(single_fields) else "gmv"
                fields = col
                N0_phi, N0_curl = np.load(f"{cache_dir}{sep}_N0{sep}{exp}{sep}{typ}{sep}N0_{fields}_{ps}_T{L_cuts[0]}-{L_cuts[1]}_P{L_cuts[2]}-{L_cuts[3]}.npy")
                if typ == "gmv": col += "_gmv"
                if convert:
                    df_phi[col] = N0_phi*4/(Ls**4)
                    df_curl[col] = N0_curl*4/(Ls**4)
                else:
                    df_phi[col] = N0_phi
                    df_curl[col] = N0_curl
            dir = f"{omegaqe.DATA_DIR}/N0/{exp}"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            df_phi.set_index("Ls", inplace=True)
            df_phi.to_csv(f"{dir}{sep}N0_phi_{ps}_T{L_cuts[0]}-{L_cuts[1]}_P{L_cuts[2]}-{L_cuts[3]}.csv", sep=" ", float_format='{:,.6e}'.format)
            df_curl.set_index("Ls", inplace=True)
            df_curl.to_csv(f"{dir}{sep}N0_curl_{ps}_T{L_cuts[0]}-{L_cuts[1]}_P{L_cuts[2]}-{L_cuts[3]}.csv", sep=" ", float_format='{:,.6e}'.format)

def save_N(exps, fields):
    columns = np.array(fields)
    Ls = np.arange(2, 5001)
    for exp in exps:
        df_N = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
        for iii, col in enumerate(columns):
            field = col
            N = np.load(f"{cache_dir}{sep}_N0{sep}{exp}{sep}exp{sep}N_{field}{field}.npy")[2:5001]
            df_N[col] = N
        dir = f"{data_dir}{sep}N0{sep}{exp}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        df_N.set_index("Ls", inplace=True)
        df_N.to_csv(f"{dir}{sep}N.csv", sep=" ", float_format='{:,.6e}'.format)

def main():
    single_fields = ["TT", "TE", "TB", "EE", "EB"]
    gmv_fields = ["TE", "EB", "TEB"]
    exps = np.array(["SO", "SO_base", "SO_goal","S4", "S4_base", "HD"])
    powerspectra = ["gradient"]
    save_N0(exps, powerspectra, single_fields, gmv_fields, (30, 3000, 30, 5000))
    exps = np.array(["SO_base", "SO_goal", "S4_base"])
    powerspectra = ["lensed"]
    gmv_fields = ["EB", "TEB"]
    save_N0(exps, powerspectra, single_fields, gmv_fields, (30, 3000, 30, 5000), convert=True)

    fields = ["T", "E", "B"]
    save_N(exps, fields)

    save_N0(["SO_goal"], ["gradient"], ["TT"], ["EB", "TEB"], (40, 3000, 40, 3000))
    save_N0(["S4_test"], ["gradient"], ["TT"], ["EB", "TEB"], (2, 4000, 2, 4000))
    save_N0(["S4_dp"], ["gradient"], ["TT"], ["EB", "TEB"], (30, 3000, 30, 5000))


if __name__ == '__main__':

    main()
