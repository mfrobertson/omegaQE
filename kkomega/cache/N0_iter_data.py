import pandas as pd
import numpy as np
import os

def save_N0(exps, powerspectra, fields, convert=False):
    columns = np.array(fields)
    Ls = np.arange(0, 5001)
    for exp in exps:
        for ps in powerspectra:
            df_phi = pd.read_csv(f'../data/N0/{exp}/N0_phi_{ps}_T30-3000_P30-5000.csv', sep=' ')
            df_curl = pd.read_csv(f'../data/N0/{exp}/N0_curl_{ps}_T30-3000_P30-5000.csv', sep=' ')
            for iii, col in enumerate(columns):
                typ = "iter"
                fields = col
                N0_phi, N0_curl = np.load(f"_N0/{exp}/{typ}/N0_{fields}_{ps}_T30-3000_P30-5000.npy")
                col += "_iter"
                if convert:
                    df_phi[col] = N0_phi[2:5001]*4/(Ls**4)
                    df_curl[col] = N0_curl[2:5001]*4/(Ls**4)
                else:
                    df_phi[col] = N0_phi[2:5001]
                    df_curl[col] = N0_curl[2:5001]
            dir = f"../data/N0/{exp}"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            df_phi.set_index("Ls", inplace=True)
            df_phi.to_csv(f"{dir}/N0_phi_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)
            df_curl.set_index("Ls", inplace=True)
            df_curl.to_csv(f"{dir}/N0_curl_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)

def main():
    fields = ["TT", "EB", "TEB"]
    exps = np.array(["S4_base"])
    powerspectra = ["lensed"]
    save_N0(exps, powerspectra, fields)


if __name__ == '__main__':

    main()
