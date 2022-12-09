import numpy as np
import matplotlib.pyplot as plt
from omegaQE import OmegaQE
from cosmology import Cosmology
from scipy import interpolate
import time
from fisher import Fisher

def get_F_L_and_C_inv_splines(fields, Lmax=5000):
    fish = Fisher()
    sample_Ls = fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=Lmax, Nells=150)
    sample_Ls, F_L, C_inv = fish.get_F_L(fields, Ls=sample_Ls, Ntheta=100, nu=353e9, return_C_inv=True)
    F_L_spline = interpolate.InterpolatedUnivariateSpline(sample_Ls, F_L)
    N_fields = np.size(fields)
    C_inv_splines = np.empty((N_fields, N_fields), dtype=interpolate.InterpolatedUnivariateSpline)
    Ls = np.arange(Lmax+1)
    for iii in range(N_fields):
        for jjj in range(N_fields):
            C_inv_ij = C_inv[iii, jjj]
            C_inv_splines[iii, jjj] = interpolate.InterpolatedUnivariateSpline(Ls[2:], C_inv_ij[2:])
    return F_L_spline, C_inv_splines


def get_omega(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline):
    qe = OmegaQE(field, N_pix=2 ** N_pix_pow, F_L_spline=F_L_spline, C_inv_spline=C_inv_spline)
    omega = qe.get_omega(Nchi, noise=True)
    return qe, omega,  qe.F_L_spline


def my_test(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline):
    qe, omega, F_L = get_omega(field, Nchi, N_pix_pow, F_L_spline, C_inv_spline)
    ps, kBins, errs = qe.fields.get_ps(omega, nBins=50)
    plt.errorbar(kBins[:], ps[:], errs[:], label=f"Npix=2**{N_pix_pow}")


field = "k"
t0 = time.time()
F_L_spline, C_inv_spline = get_F_L_and_C_inv_splines(field)
t1 = time.time()
print(f"Init F_L time : {t1 - t0:.4f}s")

t0 = time.time()
for N_pix_pow in range(7,12):
    print(N_pix_pow)
    my_test(field, 3, N_pix_pow, F_L_spline, C_inv_spline)
t1 = time.time()
print(f"Time: {t1-t0:.4f}s")

omega_Ls, omega_ps = Cosmology().get_postborn_omega_ps(10000)

plt.loglog(omega_Ls, omega_ps)
plt.loglog(omega_Ls, omega_ps/F_L_spline(omega_Ls), label="Fiducial")
plt.xlim(3e1, 4e3)
plt.ylim(1e-12,1e-5)
plt.legend()
plt.show()