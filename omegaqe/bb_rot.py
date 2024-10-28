import omegaqe
from omegaqe.fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import vector
from omegaqe.tools import getFileSep, path_exists

def _integral_prep(N_L1, N_L2, Ntheta1, Ntheta2):
    L1s = np.linspace(Lmin, Lmax, N_L1)
    L2s = np.linspace(Lmin, Lmax, N_L2)
    dTheta1 = 2 * np.pi / Ntheta1
    thetas1 = np.linspace(0, 2 * np.pi - dTheta1, Ntheta1)
    dTheta2 = 2 * np.pi / Ntheta2
    thetas2 = np.linspace(0, 2 * np.pi - dTheta2, Ntheta2)
    return L1s, thetas1, dTheta1, L2s, thetas2, dTheta2

def _BB_mixed(Ls, verbose, N_L1, N_L2, Ntheta1, Ntheta2):
    L1s, thetas1, dTheta1, L2s, thetas2, dTheta2 = _integral_prep(N_L1, N_L2, Ntheta1, Ntheta2)
    Ls_vec = vector.obj(rho=Ls, phi=0)
    cl_e = cosmo.get_lens_ps("EE", 6000)
    cl_e_spline = InterpolatedUnivariateSpline(np.arange(np.size(cl_e)), cl_e)
    I_L1 = np.zeros(np.size(L1s))
    for iii, L1 in enumerate(L1s):
        if verbose: print(f"    L1 = {L1} ({iii}/{np.size(L1s) - 1})")
        I_theta1 = np.zeros(np.size(thetas1))
        for jjj, theta1 in enumerate(thetas1):
            L1_vec = vector.obj(rho=L1, phi=theta1)
            L1p_vec = Ls_vec + L1_vec
            L1p = L1p_vec.rho
            if L1p < Lmin:
                continue
            I_fac = (L1_vec @ L1p_vec) * np.sin(2 * L1_vec.deltaphi(Ls_vec))**2 * cl_e_spline(L1)
            I_L2 = np.zeros(np.size(L2s))
            for kkk, L2 in enumerate(L2s):
                L2_vec = vector.obj(rho=L2, phi=thetas2)
                L2p_vec = Ls_vec + L2_vec
                L2p = L2p_vec.rho
                w2 = np.ones(np.size(L2p))
                w2[L2p < Lmin] = 0
                w3 = np.ones(np.size(thetas2))
                w3[(L1p_vec+L2_vec).rho < Lmin] = 0
                theta1p2 = L1p_vec.deltaphi(L2_vec)
                L_fac = (L1p**4 * L2**2) + (2*L1p**3*L2**3) + (L1p**2 * L2**4)   #Depending on convention for omega...
                bi = 8 * fish.bi.get_bispectrum("kkw", L1p, L2, theta=theta1p2, M_spline=True)/(-L_fac) #Lfac may be -ve
                I_theta2 = w2 * w3 * (L1_vec @ L2_vec) * (L1*L2p*np.sin(L2p_vec.deltaphi(L1_vec))) * bi

                I_L2[kkk] = InterpolatedUnivariateSpline(thetas2, I_theta2).integral(0, 2*np.pi-dTheta2)
            I_theta1[jjj] = InterpolatedUnivariateSpline(L2s, L2s * I_L2 * I_fac).integral(Lmin, Lmax)
        I_L1[iii] = InterpolatedUnivariateSpline(thetas1, I_theta1).integral(0, 2*np.pi-dTheta1)
    return (-2/(2*np.pi)**4) * InterpolatedUnivariateSpline(L1s, L1s * I_L1).integral(Lmin, Lmax)

def _BB_corr(Ls, verbose, N_L1, N_L2, Ntheta1, Ntheta2):
    L1s, thetas1, dTheta1, L2s, thetas2, dTheta2 = _integral_prep(N_L1, N_L2, Ntheta1, Ntheta2)
    pass


def spectra(typ, Ls, verbose=False, N_L1=100, N_L2=100, Ntheta1=10, Ntheta2=10):
    global fish, cosmo, Lmin, Lmax
    fish = Fisher(setup_bispectra=True)
    cosmo = fish.power.cosmo
    Lmin = 2
    Lmax = 5000
    if typ == "mixed":
        return _BB_mixed(Ls, verbose, N_L1, N_L2, Ntheta1, Ntheta2)
    if typ == "corr":
        return _BB_corr(Ls, verbose, N_L1, N_L2, Ntheta1, Ntheta2)
    raise ValueError(f"Type {typ} not supported")