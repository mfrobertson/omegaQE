import numpy as np

import omegaqe
from omegaqe.modecoupling import Modecoupling
from omegaqe.tools import getFileSep
from scipy.interpolate import InterpolatedUnivariateSpline
import vector


def _get_postborn_omega_ps(Ls, M_path, Nell_prim, Ntheta, M_ellmax, M_Nell, cmb):
    Lmin = np.min(Ls)
    Lmax = 2*np.max(Ls)
    sep = getFileSep()
    mode_typ = "ww" if cmb else "rr"
    ells_sample = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "ells.npy")
    M = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "M.npy")
    M_spline = Modecoupling().spline(ells_sample, M)
    dTheta = np.pi / Ntheta
    thetas = np.linspace(dTheta, np.pi, Ntheta, dtype=float)
    floaty = Lmax / 1000
    samp1 = np.arange(Lmin, floaty * 10, 1)
    samp2 = np.logspace(1, 3, Nell_prim - np.size(samp1)) * floaty
    Lprims = np.concatenate((samp1, samp2))
    N_Ls = np.size(Ls)
    I = np.zeros(N_Ls)
    if N_Ls == 1:
        Ls = np.array([Ls])
    for iii, L in enumerate(Ls):
        L_vec = vector.obj(rho=L, phi=0)
        I_tmp = np.zeros(np.size(Lprims))
        for jjj, Lprim in enumerate(Lprims):
            Lprim_vec = vector.obj(rho=Lprim, phi=thetas)
            Lprimprim_vec = L_vec - Lprim_vec
            Lprimprims = Lprimprim_vec.rho
            I_tmp[jjj] = np.sum(
                2 * Lprim * dTheta * (L * Lprim * np.sin(thetas)) ** 2 * (Lprim_vec @ Lprimprim_vec) ** 2 / (
                            (Lprim) ** 4 * (Lprimprims) ** 4) * M_spline.ev(Lprim, Lprimprims))
        I[iii] = InterpolatedUnivariateSpline(Lprims, I_tmp).integral(Lmin, Lmax)
    return 4 * I / ((2 * np.pi) ** 2)


def omega_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True):
    """

    Parameters
    ----------
    ells
    Nell_prim
    Ntheta

    Returns
    -------

    """
    return _get_postborn_omega_ps(ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb)