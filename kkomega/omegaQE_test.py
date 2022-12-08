import numpy as np
import matplotlib.pyplot as plt
from omegaQE import OmegaQE
from cosmology import Cosmology
from scipy import stats
import time

def get_ps(omega, qe):
    nBins=50
    ps = np.real(np.conjugate(omega) * omega)
    ps[1:, 0] /= 2
    ps[1:, -1] /= 2
    means, bin_edges, binnumber = stats.binned_statistic(qe.L_map.flatten(), ps.flatten(), 'mean', bins=nBins)
    binSeperation = bin_edges[1]
    kBins = np.asarray([bin_edges[i] - binSeperation / 2 for i in range(1, len(bin_edges))])
    return means, kBins

def get_omega(field, Nchi, N_pix_power):
    qe = OmegaQE(field, N_pix=2 ** N_pix_power)
    omega = qe.get_omega(Nchi, noise=True)
    return qe, omega,  qe.F_L_spline

t0 = time.time()
qe, omega_k_7, F_L_k = get_omega("k", 20, 7)
ps_7, kBins_7 = get_ps(omega_k_7, qe)
print(f"mean 7: {np.mean(omega_k_7)}")
qe, omega_k_8, _ = get_omega("k", 20, 8)
ps_8, kBins_8 = get_ps(omega_k_8, qe)
print(f"mean 8: {np.mean(omega_k_8)}")
qe, omega_k_9, _ = get_omega("k", 20, 9)
ps_9, kBins_9 = get_ps(omega_k_9, qe)
print(f"mean 9: {np.mean(omega_k_9)}")
qe, omega_k_10, _ = get_omega("k", 20, 10)
ps_10, kBins_10 = get_ps(omega_k_10, qe)
print(f"mean 10: {np.mean(omega_k_10)}")
t1 = time.time()
print(f"Time: {t1-t0}")



omega_Ls, omega_ps = Cosmology().get_postborn_omega_ps(10000)

plt.loglog(kBins_7[:], ps_7[:], label="Npix=2**7")
plt.loglog(kBins_8[:], ps_8[:], label="Npix=2**8")
plt.loglog(kBins_9[:], ps_9[:], label="Npix=2**9")
plt.loglog(kBins_10[:], ps_10[:], label="Npix=2**10")
plt.loglog(omega_Ls, omega_ps)
plt.loglog(omega_Ls, omega_ps/F_L_k(omega_Ls))
plt.xlim(3e1, 4e3)
plt.ylim(1e-12,1e-5)
plt.legend()
plt.show()