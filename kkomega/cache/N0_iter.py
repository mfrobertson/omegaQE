import numpy as np
import tools
import sys
import os
from plancklens import utils, n0s
import plancklens
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from covariance import Covariance

def save(N0, fields, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax, ext):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    gmv_str = "iter_ext" if ext else "iter"
    folder = current_folder + sep + "_N0" + sep + exp + sep + gmv_str
    filename_N0 = f"N0_{fields}_T{T_Lmin}-{T_Lmax}_P{P_Lmin}-{P_Lmax}.npy"
    print(f"Saving {filename_N0} at {folder}")
    tools.save_array(folder, filename_N0, N0)

def get_qe_key(fields, gmv):
    if not gmv:
        if fields != "TT":
            raise ValueError(f"Cannot get iterative N0 for single qe field pair {fields}. Only TT.")
        return "ptt"
    elif gmv:
        if fields == "EB":
            return "p_p"
        elif fields == "TEB":
            return "p"
        else:
            raise ValueError(f"Cannot get iterative N0 for minimum variance combination of {fields}. Only EB or TEB.")

def _get_cmb_gaussian_N(typ, ellmax, exp):
        if typ[0] != typ[1]:
            return np.zeros(ellmax + 1)
        sep = tools.getFileSep()
        dir = f'../data{sep}N0{sep}{exp}{sep}'
        N = np.array(pd.read_csv(dir + f'N.csv', sep=' ')[typ[0]])[:ellmax + 1]
        N = np.concatenate((np.zeros(2), N))
        return N

def _deconstruct_noise_curve(typ, exp, beam, Lmax):
    Lmax_data = 5000
    N = _get_cmb_gaussian_N(typ, Lmax_data, exp)
    T_cmb = 2.7255
    arcmin_to_rad = np.pi / 180 / 60
    Ls = np.arange(np.size(N))
    beam *= arcmin_to_rad
    deconvolve_beam = np.exp(Ls * (Ls + 1) * beam ** 2 / (8 * np.log(2)))
    n = np.sqrt(N / deconvolve_beam) * T_cmb / 1e-6 / arcmin_to_rad
    # plancklens_ellmax_sky = 5000
    plancklens_ellmax_sky = Lmax
    Ls_sample = np.arange(np.size(N))
    Ls = np.arange(plancklens_ellmax_sky + 1)
    return InterpolatedUnivariateSpline(Ls_sample, n)(Ls)


def _noise_args(delta_T, beam, Lmax):
    lensit_ellmax_sky = 6000
    lensit_ellmax_sky = Lmax
    if delta_T is None or beam is None:
        beam = 0
        nT = _deconstruct_noise_curve("TT", exp, beam, lensit_ellmax_sky)
        nP = _deconstruct_noise_curve("EE", exp, beam, lensit_ellmax_sky)
        return nT, nP, beam
    nT = np.ones(lensit_ellmax_sky + 1) * delta_T
    nP = np.sqrt(2) * nT
    return nT, nP, beam

def get_N_dict(exp, delta_T, beam, Lmax):
    if exp == "SO_base" or exp == "S4_base":
        delta_T = None
        beam = None
    N_dict = {}
    os.chdir("../")
    for fields in ["TT", "EE", "BB"]:
        N = cov.noise.get_cmb_gaussian_N(fields, deltaT=delta_T, beam=beam, ellmax=Lmax, exp=exp) * (2.7255 * 1e6) ** 2
        N_dict[fields.lower()] = np.concatenate((np.array([N[0], N[0]]), N))
    os.chdir("cache/")
    return N_dict

def main(exp, fields, gmv, iters, T_Lmin, T_Lmax, P_Lmin, P_Lmax, ext):
    global cov
    cov = Covariance()
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    qe_key = get_qe_key(fields, gmv)
    delta_T, beam = cov.noise.get_noise_args(exp)
    delta_T, delta_P, beam = _noise_args(delta_T, beam, np.max([T_Lmax, P_Lmax]))
    rho_sqd_ext = cov.get_total_tracer_corr("gI", 5000)**2 if ext else 0
    for key in cls_unl.keys():
        cls_unl[key] = cls_unl[key][:5001]
    N0_iter = n0s.get_N0_iter(qe_key, delta_T, delta_P, beam, cls_unl, {'t': T_Lmin, 'e': P_Lmin, 'b': P_Lmin}, {'t': T_Lmax, 'e': P_Lmax, 'b': P_Lmax}, lmax_qlm=5000, itermax=iters, ret_delcls=False, ret_curl=True, rho_sqd_ext=rho_sqd_ext)
    N0 = (N0_iter[0][iters], N0_iter[2][iters])
    save(N0, fields, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax, ext)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 9:
        raise ValueError("Must supply arguments: exp fields gmv iters T_Lmin T_Lmax P_Lmin P_Lmax ext")
    exp = str(args[0])
    fields = str(args[1])
    gmv = tools.parse_boolean(args[2])
    iters = int(args[3])
    T_Lmin = int(args[4])
    T_Lmax = int(args[5])
    P_Lmin = int(args[6])
    P_Lmax = int(args[7])
    ext = tools.parse_boolean(args[8])
    main(exp, fields, gmv, iters, T_Lmin, T_Lmax, P_Lmin, P_Lmax, ext)