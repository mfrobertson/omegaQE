import numpy as np
import matplotlib.pyplot as plt
from omegaQE import OmegaQE
from cosmology import Cosmology
import time
from fisher import Fisher
from scipy.interpolate import InterpolatedUnivariateSpline

def get_F_L_and_C_inv_splines(fields, L_max_map=5000, L_min_cut=30, L_max_cut=3000):
    fish = Fisher()
    sample_Ls = fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=L_max_map, Nells=200)
    sample_Ls, F_L, C_inv = fish.get_F_L(fields, Ls=sample_Ls, Nell2=1000, Ntheta=1000, nu=353e9, return_C_inv=True, Lmin=L_min_cut, Lmax=L_max_cut)
    F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
    N_fields = np.size(fields)
    C_inv_splines = np.empty((N_fields, N_fields), dtype=InterpolatedUnivariateSpline)
    Ls = np.arange(L_max_map+1)
    for iii in range(N_fields):
        for jjj in range(N_fields):
            C_inv_ij = C_inv[iii, jjj]
            C_inv_ij[L_max_map+1:] = 0
            C_inv_ij[:L_min_cut] = 0
            C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls, C_inv_ij)
    return F_L_spline, C_inv_splines


def get_omega(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline):
    qe = OmegaQE(field, N_pix=2 ** N_pix_pow, F_L_spline=F_L_spline, C_inv_spline=C_inv_spline, Lmin=30, Lmax=3000, Lmax_map=5000)
    omega = qe.get_omega(Nchi)
    return qe, omega,  qe.F_L_spline


def my_test(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline):
    qe, omega, F_L = get_omega(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline)
    ps, kBins, errs = qe.fields.get_ps(omega, nBins=50, kmin=30, kmax=3000)
    plt.errorbar(kBins[:], ps[:], errs[:], label=f"Npix=2**{N_pix_pow}")


field = "k"
t0 = time.time()
F_L_spline, C_inv_spline = get_F_L_and_C_inv_splines(field)
t1 = time.time()
print(f"Init F_L time : {t1 - t0:.4f}s")

t0 = time.time()
for N_pix_pow in range(9,13):
    print(N_pix_pow)
    my_test(field, 100, N_pix_pow, F_L_spline, C_inv_spline)
t1 = time.time()
print(f"Time: {t1-t0:.4f}s")

omega_Ls, omega_ps = Cosmology().get_postborn_omega_ps(10000)
omega_spline = InterpolatedUnivariateSpline(omega_Ls, omega_ps)
Ls = np.arange(30, 3000)
plt.loglog(Ls, (2 * np.pi) ** 2 * omega_spline(Ls)/F_L_spline(Ls), label="Fiducial")
plt.xlim(3e1, 3e3)
plt.ylim(1e-8,4e-6)
plt.legend()
plt.show()
