import pandas as pd
import numpy as np
import os
import omegaqe
import omegaqe.tools as tools

cache_dir = omegaqe.CACHE_DIR
data_dir = omegaqe.DATA_DIR
sep = tools.getFileSep()


def save_N0(exps, powerspectra, fields, convert=False):
    columns = np.array(fields)
    for exp in exps:
        for ps in powerspectra:
            df_phi = pd.read_csv(f'{data_dir}{sep}N0{sep}{exp}{sep}N0_phi_{ps}_T30-3000_P30-5000.csv', sep=' ')
            df_curl = pd.read_csv(f'{data_dir}{sep}N0{sep}{exp}{sep}N0_curl_{ps}_T30-3000_P30-5000.csv', sep=' ')
            for iii, col in enumerate(columns):
                for typ in ["iter", "iter_ext"]:
                    fields = col
                    N0_phi, N0_curl = np.load(f"{cache_dir}_N0{sep}{exp}{sep}{typ}{sep}N0_{fields}_T30-3000_P30-5000.npy")
                    col_name = col + "_" + typ
                    if convert:
                        Ls = np.arange(2, 5001)
                        df_phi[col_name] = N0_phi[2:5001]*4/(Ls**4)
                        df_curl[col_name] = N0_curl[2:5001]*4/(Ls**4)
                    else:
                        df_phi[col_name] = N0_phi[2:5001]
                        df_curl[col_name] = N0_curl[2:5001]
            dir = f"{data_dir}{sep}N0{sep}{exp}"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            df_phi.set_index("Ls", inplace=True)
            df_phi.to_csv(f"{dir}{sep}N0_phi_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)
            df_curl.set_index("Ls", inplace=True)
            df_curl.to_csv(f"{dir}{sep}N0_curl_{ps}_T30-3000_P30-5000.csv", sep=" ", float_format='{:,.6e}'.format)

def main():
    fields = ["TT", "EB", "TEB"]
    exps = np.array(["SO_base", "SO_goal", "S4_base", "S4_dp"])
    powerspectra = ["gradient"]
    save_N0(exps, powerspectra, fields)


if __name__ == '__main__':

    main()
