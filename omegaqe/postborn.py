import numpy as np

import omegaqe
from omegaqe.modecoupling import Modecoupling
from omegaqe.tools import getFileSep
from scipy.interpolate import InterpolatedUnivariateSpline
import vector

mode = Modecoupling()


def _get_modecoupling(M_path, M_ellmax, M_Nell, cmb, zmin, zmax, Lmax):
    if (zmin == 0 and zmax) is None:
        sep = getFileSep()
        mode_typ = "ww" if cmb else "rr"
        ells_sample = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "ells.npy")
        M = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "M.npy")
        return mode.spline(ells_sample, M)
    mode_typ = "kk" if cmb else "ss"
    return mode.spline(ells_sample=mode.generate_sample_ells(Lmax, 100), typ=mode_typ, star=False, zmin=zmin, zmax=zmax)


def _get_postborn_omega_ps(Ls, M_path, Nell_prim, Ntheta, M_ellmax, M_Nell, cmb, zmin=0, zmax=None):
    Lmin = np.min(Ls)
    Lmax = 2*np.max(Ls)
    M_spline = _get_modecoupling(M_path, M_ellmax, M_Nell, cmb, zmin, zmax, Lmax)
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


def omega_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True, zmin=0, zmax=None):
    """

    Parameters
    ----------
    ells
    Nell_prim
    Ntheta

    Returns
    -------

    """
    return _get_postborn_omega_ps(ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb, zmin, zmax)