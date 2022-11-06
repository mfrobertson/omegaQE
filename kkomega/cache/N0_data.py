import pandas as pd
import numpy as np
import os

def save_N0(exps, powerspectra, single_fields, gmv_fields, convert=False):
    columns = np.array(single_fields + gmv_fields)
    Ls = np.arange(2, 5001)
    for exp in exps:
        for ps in powerspectra:
            df_phi = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
            df_curl = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
            for iii, col in enumerate(columns):
                typ = "single" if iii < np.size(single_fields) else "gmv"
                fields = col
                N0_phi, N0_curl = np.load(f"_N0/{exp}/{typ}/N0_{fields}_{ps}_T30-3000_P30-5000.npy")
                if typ == "gmv": col += "_gmv"
                if convert:
                    df_phi[col] = N0_phi*4/(Ls**4)
                    df_curl[col] = N0_curl*4/(Ls**4)
                else:
                    df_phi[col] = N0_phi
                    df_curl[col] = N0_curl
            dir = f"../data/N0/{exp}"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            df_phi.set_index("Ls", inplace=True)
            df_phi.to_csv(f"{dir}/N0_phi_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)
            df_curl.set_index("Ls", inplace=True)
            df_curl.to_csv(f"{dir}/N0_curl_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)

def save_N(exps, fields):
    columns = np.array(fields)
    Ls = np.arange(2, 5001)
    for exp in exps:
        df_N = pd.DataFrame(data=np.hstack((Ls[:, None])), columns=["Ls"])
        for iii, col in enumerate(columns):
            field = col
            N = np.load(f"_N0/{exp}/exp/N_{field}{field}.npy")[2:5001]
            df_N[col] = N
        dir = f"../data/N0/{exp}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        df_N.set_index("Ls", inplace=True)
        df_N.to_csv(f"{dir}/N.csv", sep=" ", float_format='{:,.6e}'.format)

def main():
    single_fields = ["TT", "TE", "TB", "EE", "EB"]
    gmv_fields = ["TE", "EB", "TEB"]
    exps = np.array(["SO", "SO_base", "SO_goal","S4", "S4_base", "HD"])
    powerspectra = ["gradient"]
    save_N0(exps, powerspectra, single_fields, gmv_fields)
    exps = np.array(["SO_base", "SO_goal", "S4_base"])
    powerspectra = ["lensed"]
    gmv_fields = ["EB", "TEB"]
    save_N0(exps, powerspectra, single_fields, gmv_fields, convert=True)
    fields = ["T", "E", "B"]
    save_N(exps, fields)


if __name__ == '__main__':

    main()
