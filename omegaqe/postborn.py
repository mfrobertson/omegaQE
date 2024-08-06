import numpy as np
import omegaqe
from omegaqe.modecoupling import Modecoupling
from omegaqe.tools import getFileSep
from scipy.interpolate import InterpolatedUnivariateSpline
import vector


def _get_modecoupling(M_path, M_ellmax, M_Nell, cmb, zmin, zmax, Lmax, powerspectra):
    matter_PK = powerspectra.matter_PK if powerspectra is not None else None
    mode = Modecoupling(powerspectra=powerspectra, matter_PK=matter_PK)
    if zmin == 0 and zmax is None:
        sep = getFileSep()
        mode_typ = "ww" if cmb else "rr"
        ells_sample = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "ells.npy")
        M = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "M.npy")
        return mode.spline(ells_sample, M)
    print("Generating new M_ww... ")
    mode_typ = "kk" if cmb else "ss"
    return mode.spline(ells_sample=mode.generate_sample_ells(M_ellmax, M_Nell), typ=mode_typ, star=False, zmin=zmin, zmax=zmax)

def _get_postborn_ps(typ, Ls, M_path, Nell_prim, Ntheta, M_ellmax, M_Nell, cmb, zmin=0, zmax=None, powerspectra=None):
    Lmin = np.min(Ls)
    Lmax = 2*np.max(Ls)
    M_spline = _get_modecoupling(M_path, M_ellmax, M_Nell, cmb, zmin, zmax, Lmax, powerspectra)
    dTheta = np.pi / Ntheta
    thetas = np.linspace(dTheta, np.pi, Ntheta, dtype=float)
    Lprims = np.geomspace(Lmin, Lmax, Nell_prim)
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
            w = np.ones(np.shape(Lprimprims))
            w[Lprimprims < Lmin] = 0
            w[Lprimprims > Lmax] = 0
            I_tmp[jjj] = np.sum(_get_integrand(typ, w, L_vec, Lprim_vec, Lprimprim_vec, thetas, dTheta, M_spline))
        I[iii] = InterpolatedUnivariateSpline(Lprims, I_tmp).integral(Lmin, Lmax)
    return 4 * I / ((2 * np.pi) ** 2)

def _get_integrand(typ, w, L_vec, Lprim_vec, Lprimprim_vec, thetas, dTheta, M_spline):
    L = L_vec.rho
    Lprim = Lprim_vec.rho
    Lprimprims = Lprimprim_vec.rho
    if typ == "omega":
        return 2 * w * Lprim * dTheta * (L * Lprim * np.sin(thetas)) ** 2 * (Lprim * Lprimprims * np.cos(Lprimprim_vec.deltaphi(Lprim_vec))) ** 2 / ((Lprim) ** 4 * (Lprimprims) ** 4) * M_spline.ev(Lprim, Lprimprims)
    if typ == "len_len_kappa":
        return 2 * w * Lprim * dTheta * (L_vec * Lprimprims * np.cos(Lprimprim_vec.deltaphi(L_vec))) ** 2 * (Lprim * Lprimprims * np.cos(Lprimprim_vec.deltaphi(Lprim_vec))) ** 2 / ((Lprim) ** 4 * (Lprimprims) ** 4) * M_spline.ev(Lprimprims, Lprim)
    if typ == "ray_def_kappa":
        return -2 * Lprim * dTheta * (L * Lprim * np.cos(thetas)) ** 2 / ((Lprim) ** 4) * M_spline.ev(L, Lprim)
    if typ == "pb_kappa":
        return (2 * w * Lprim * dTheta * (L_vec * Lprimprims * np.cos(Lprimprim_vec.deltaphi(L_vec))) ** 2 * (Lprim * Lprimprims * np.cos(Lprimprim_vec.deltaphi(Lprim_vec))) ** 2 / ((Lprim) ** 4 * (Lprimprims) ** 4) * M_spline.ev(Lprimprims, Lprim)) - (2 * Lprim * dTheta * (L * Lprim * np.cos(thetas)) ** 2 / ((Lprim) ** 4) * M_spline.ev(L, Lprim))
    raise ValueError(f"Type: {typ} not recognised")

def omega_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True, zmin=0, zmax=None, powerspectra=None):
    return _get_postborn_ps("omega", ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb, zmin, zmax, powerspectra)

def len_len_kappa_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True, zmin=0, zmax=None, powerspectra=None):
    return _get_postborn_ps("len_len_kappa", ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb, zmin, zmax, powerspectra)

def ray_def_kappa_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True, zmin=0, zmax=None, powerspectra=None):
    return _get_postborn_ps("ray_def_kappa", ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb, zmin, zmax, powerspectra)

def postborn_kappa_ps(ells, M_path=f"{omegaqe.CACHE_DIR}/_M", Nell_prim=1000, Ntheta=500, cmb=True, zmin=0, zmax=None, powerspectra=None):
    return _get_postborn_ps("pb_kappa", ells, M_path, Nell_prim, Ntheta, 10000, 200, cmb, zmin, zmax, powerspectra)