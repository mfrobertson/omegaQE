import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from omegaqe.fisher import Fisher
from omegaqe.cosmology import Cosmology
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def kappa_cross_Cov(typs, Ls, F_L, A_tilde, NL1s=1000, tracer_Lmin=30, tracer_Lmax=3000):
    def triangle_lims(A, B):
        third_side_min = np.max([np.abs(A-B), tracer_Lmin])
        third_side_max = np.min([A+B, tracer_Lmax])
        return np.floor(third_side_min), np.ceil(third_side_max)

    def get_typs_combos(typs):
        all_combos = typs[:, None] + typs[None, :]
        return all_combos.flatten()

    cosmo = Cosmology("DEMNUnii")
    fish = Fisher(exp="ACT", cosmology=cosmo, setup_bispectra=True)
    fish.covariance.noise.n = 7
    NLs = np.size(Ls)
    Cov = np.zeros((NLs, NLs))
    typs = np.char.array(list(typs))
    typ_combs = get_typs_combos(typs)
    nu = 353e9
    C_inv = fish.covariance.get_C_inv(typs, int(np.ceil(np.max(Ls))), nu)
    cl_kappa = fish.covariance.power.get_kappa_ps(Ls)
    N0 = fish.covariance.noise.get_N0("kappa", int(np.ceil(np.max(Ls))))
    for ip in typ_combs:
        i = ip[0]
        p = ip[1]
        cinv_idx_i = np.where(typs == i)[0][0]
        cinv_idx_p = np.where(typs == p)[0][0]
        cinv_ip_spline = fish._interpolate(C_inv[cinv_idx_i][cinv_idx_p])
        for iii, L in enumerate(Ls):
            for jjj in np.arange(iii, NLs):
                if iii == jjj:
                    continue
                Lp = Ls[jjj]
                if Lp < tracer_Lmin or Lp > tracer_Lmax:
                    continue
                Lmin, Lmax = triangle_lims(L, Lp)
                L1s = np.linspace(Lmin, Lmax, NL1s)
                bi1 = fish.bi.get_bispectrum(f"k{i}k", Lp, L1s, L, M_spline=True, one_perm=True)
                if i == p:
                    bi2 = bi1
                else:
                    bi2 = fish.bi.get_bispectrum(f"k{p}k", Lp, L1s, L, M_spline=True, one_perm=True)
                I = L1s * bi1 * bi2 * cinv_ip_spline(L1s)
                Cov_ij = InterpolatedUnivariateSpline(L1s, I).integral(Lmin, Lmax) / F_L[iii] / F_L[jjj] / (2 * np.pi)
                # L_facs = (2*L + 2*Lp + 2)/((2*L + 1) * (2*Lp + 1))
                L_facs = 1 / ((2 * L + 1) * (2 * Lp + 1))
                Cov[iii, jjj] += L_facs * Cov_ij - 1
                Cov[jjj, iii] += L_facs * Cov_ij - 1
    for iii, L in enumerate(Ls):
        cov_kk = cl_kappa[iii] + N0[L]
        Cov[iii, iii] = ((cov_kk * A_tilde[iii]) + 1) / ((2 * L) + 1)
    return Ls, Cov


def main(typs, Ls, F_L, A_tilde):
    t0 = time()
    Ls, Cov = kappa_cross_Cov(typs, Ls, F_L, A_tilde, NL1s=10)
    t1 = time()
    print(f"Done in {t1 - t0:.2f} seconds")
    print(Cov)
    # plt.imshow(np.abs(Cov), norm=LogNorm(1e-3,1e6), extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))
    # plt.colorbar()
    # plt.show()

    std_devs = np.sqrt(np.diag(Cov))

    std_matrix = np.outer(std_devs, std_devs)

    # plt.imshow(Cov / std_matrix, vmin=-0.4, vmax=1, extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))
    plt.imshow(np.abs(Cov / std_matrix), norm=LogNorm(1e-2,1e0), extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))
    plt.colorbar()
    plt.show()




if __name__ == '__main__':
    typs = "kg"
    Ls = np.arange(2, 5000, 100)
    F_L_Ls = np.load("../../omegaqeNBs/_kappa_F_L_results_1perm/kg/ACT/gmv/TEB/30_3000/1_2000/Ls_k.npy")
    F_L = np.load("../../omegaqeNBs/_kappa_F_L_results_1perm/kg/ACT/gmv/TEB/30_3000/1_2000/F_L_k.npy")
    F_L_spline = InterpolatedUnivariateSpline(F_L_Ls, F_L)
    F_L_pb = np.load("../../omegaqeNBs/_kappa_F_L_results_1perm/kg/ACT/gmv/TEB/30_3000/1_2000/F_L_k_pB.npy")
    A_tilde = F_L_pb / (F_L**2)
    A_tilde_spline = InterpolatedUnivariateSpline(F_L_Ls, A_tilde)
    main(typs, Ls, F_L_spline(Ls), A_tilde_spline(Ls))
