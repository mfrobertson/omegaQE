import numpy as np
from scipy.optimize import curve_fit
import omegaqe
import omegaqe.tools as tools
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from omegaqe.powerspectra import Powerspectra
from omegaqe.noise import Noise

data_dir = omegaqe.DATA_DIR
sep = tools.getFileSep()

_power = Powerspectra()
_noise = Noise()


def _get_Cl_cib_353(ls, b_c):
    N_cib = _noise.get_cib_shot_N(353e9) * 1e12
    #N_cib=[0]
    return _power.get_cib_ps(ls, Nchi=1000, nu=353e9, bias=b_c) + N_cib[0]
def _get_Cl_cib_545(ls, b_c):
    N_cib = _noise.get_cib_shot_N(545e9) * 1e12
    #N_cib = [0]
    return _power.get_cib_ps(ls, Nchi=1000, nu=545e9, bias=b_c) + N_cib[0]
def _get_Cl_cib_857(ls, b_c):
    N_cib = _noise.get_cib_shot_N(857e9) * 1e12
    #N_cib = [0]
    return _power.get_cib_ps(ls, Nchi=1000, nu=857e9, bias=b_c) + N_cib[0]

def get_data(file_path):
    return pd.read_csv(file_path)

def get_amplitude(func, Ls, data, errs, p0, typ):
    popt, pcov = curve_fit(func, Ls, data, p0=p0, sigma=errs, absolute_sigma=True)
    print(f"For {typ}; best fit amplitude = {popt[0]*1e-6} MJy/sr, with error = {np.sqrt(pcov[0][0])*1e-6}")
    return popt[0]*1e-6


def main():
    file_path = f"{data_dir}{sep}planck_cib{sep}cib_power_spectrum.csv"
    data = get_data(file_path)
    b_c_353 = get_amplitude(_get_Cl_cib_353, data["Ls"][2:-1], data["353"][2:-1], data["353_error"][2:-1], 1e-64, "353e9")
    b_c_545 = get_amplitude(_get_Cl_cib_545, data["Ls"][2:], data["545"][2:], data["545_error"][2:], 1e-64, "545e9")
    b_c_857 = get_amplitude(_get_Cl_cib_857, data["Ls"][2:], data["857"][2:], data["857_error"][2:], 1e-65, "857e9")
    np.save(f"{data_dir}{sep}planck_cib{sep}b_c.npy", [b_c_353, b_c_545, b_c_857])


if __name__ == "__main__":
    main()