import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from omegaqe.fisher import Fisher
from omegaqe.cosmology import Cosmology
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from omegaqe.postborn import pb22_kappa_ps

def kappa_cross_Cov(typs, Ls, F_L, NL1s=1000, F_L_tot=None, tracer_Lmin=30, tracer_Lmax=3000, diag=True):
    def triangle_lims(A, B):
        third_side_min = np.max([np.abs(A-B), tracer_Lmin])
        third_side_max = np.min([A+B, tracer_Lmax])
        return np.floor(third_side_min), np.ceil(third_side_max)

    def get_typs_combos(typs):
        all_combos = typs[:, None] + typs[None, :]
        return all_combos.flatten()

    cosmo = Cosmology("DEMNUnii")
    cosmo.b1=0
    fish = Fisher(exp="ACT", cosmology=cosmo, setup_bispectra=False) # should only be true for testing
    fish.covariance.noise.n = 7

    print("calc cl_kappa_pB22")
    ells_samp = np.geomspace(1, 5000, 200)
    cl_kappa_pB_tmp = pb22_kappa_ps(ells_samp, powerspectra=fish.covariance.power)
    cl_kappa_pB = InterpolatedUnivariateSpline(ells_samp, cl_kappa_pB_tmp)(Ls)
    print("finished cl_kappa_pB22")

    NLs = np.size(Ls)
    Cov = np.zeros((NLs, NLs))
    typs = np.char.array(list(typs))
    typ_combs = get_typs_combos(typs)
    nu = 353e9
    C_inv_tmp = fish.covariance.get_C_inv(typs, 5000, nu)
    C = np.empty((np.size(typs), np.size(typs)), dtype=InterpolatedUnivariateSpline)
    C_inv = np.empty((np.size(typs), np.size(typs)), dtype=InterpolatedUnivariateSpline)
    for iii in np.arange(np.size(typs)):
        for jjj in np.arange(iii, np.size(typs)):
            typ_i = typs[iii]
            typ_j = typs[jjj]
            C_ij = fish.covariance.get_Cov(typ_i+typ_j, 5000)
            C[iii][jjj] = InterpolatedUnivariateSpline(np.arange(np.size(C_ij)), C_ij)
            C[jjj][iii] = C[iii][jjj]

            C_inv_ij = C_inv_tmp[iii][jjj]
            C_inv[iii][jjj] = InterpolatedUnivariateSpline(np.arange(np.size(C_inv_ij)),C_inv_ij)
            C_inv[jjj][iii] = C_inv[iii][jjj]
    cinv_idx_k = np.where(typs == "k")[0][0]
    if F_L_tot is not None:
        cl_kappa_cross = cl_kappa_pB*F_L_tot/F_L
    else:
        cl_kappa_cross = np.zeros(np.size(F_L))
    Ntheta=20
    # thetas, dTheta = fish._get_thetas(Ntheta, max_angle=2*np.pi)
    for iii, L in enumerate(Ls):
        print(f"{iii} out of {np.size(Ls)}")
        jjj_max = iii + 1 if diag else NLs
        for jjj in np.arange(iii, jjj_max):
            Lp = Ls[jjj]
            L1_min, L1_max = triangle_lims(Lp, L)
            L1s = np.arange(L1_min, L1_max+1)
            L_fac = 2 * L + 1
            Lp_fac = 2 * Lp + 1
            L_facs = 1 / (L_fac * Lp_fac)
            for ip in typ_combs:
                i = ip[0]
                p = ip[1]
                cinv_idx_i = np.where(typs == i)[0][0]
                cinv_idx_p = np.where(typs == p)[0][0]
                cinv_ip = C_inv[cinv_idx_i][cinv_idx_p](L1s)
                for jq in typ_combs:
                    j = jq[0]
                    q = jq[1]
                    cinv_idx_j = np.where(typs == j)[0][0]
                    cinv_idx_q = np.where(typs == q)[0][0]
                    bi_ij = fish.bi.get_bispectrum(f"{i}{j}k", L1s, Lp, L, M_spline=True, one_perm=True)
                    bi_pq = fish.bi.get_bispectrum(f"{p}{q}k", L1s, L, Lp, M_spline=True, one_perm=True)
                    for ur in typ_combs:
                        u = ur[0]
                        r = ur[1]
                        cinv_idx_u = np.where(typs == u)[0][0]
                        cinv_idx_r = np.where(typs == r)[0][0]
                        cinv_jr = C_inv[cinv_idx_j][cinv_idx_r](Lp)
                        cinv_qu = C_inv[cinv_idx_q][cinv_idx_u](L)
                        c_rk = C[cinv_idx_r][cinv_idx_k](Lp)
                        c_uk = C[cinv_idx_u][cinv_idx_k](L)
                        Cov_ij = np.sum(bi_ij * bi_pq * cinv_ip * cinv_jr * cinv_qu * c_rk * c_uk)
                        Cov_ij *= L_facs * cl_kappa_pB[iii] * cl_kappa_pB[jjj] / F_L[iii] / F_L[jjj] / (2 * np.pi)**2
                        Cov[iii, jjj] += Cov_ij
                        if iii != jjj:
                            Cov[jjj, iii] += Cov_ij
            # Cov[iii, jjj] -= cl_kappa_cross[iii] * cl_kappa_cross[jjj]
            # if iii  != jjj:
            #     Cov[jjj, iii] -= cl_kappa_cross[iii] * cl_kappa_cross[jjj]
    for iii, L in enumerate(Ls):
        cov_kk = C[cinv_idx_k][cinv_idx_k](L)
        Cov[iii, iii] += ((cov_kk * cl_kappa_pB[iii]**2/ F_L[iii]) + cl_kappa_cross[iii]**2)/ ((2 * L) + 1)
    return Ls, Cov


def main(typs, Ls, F_L, NL1s, F_L_tot):
    t0 = time()
    Ls, Cov = kappa_cross_Cov(typs, Ls, F_L, NL1s, F_L_tot, diag=False)
    t1 = time()
    print(f"Done in {t1 - t0:.2f} seconds")

    np.save(f"../../omegaqeNBs/_kappa_cross_cov_{np.size(Ls)}_2.npy", Cov)
    np.save(f"../../omegaqeNBs/_kappa_cross_cov_{np.size(Ls)}_Ls_2.npy", Ls)


    std_devs = np.sqrt(np.diag(Cov))

    std_matrix = np.outer(std_devs, std_devs)


    # plt.imshow(Cov / std_matrix, vmin=-0.4, vmax=1, extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))
    plt.imshow(np.abs(Cov / std_matrix), norm=LogNorm(1e-2,1e0), extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))
    # plt.imshow(np.abs(Cov / std_matrix), extent=(Ls[0], Ls[-1], Ls[-1], Ls[0]))

    plt.colorbar()
    plt.show()




if __name__ == '__main__':
    typs = "k"
    Ls = np.arange(30, 3001, 1)
    F_L_Ls = np.load(f"../../omegaqeNBs/_kappa_F_L_results/{typs}/ACT/gmv/TEB/30_3000/1_2000/Ls_k_pB.npy")
    F_L_tmp = np.load(f"../../omegaqeNBs/_kappa_F_L_results/{typs}/ACT/gmv/TEB/30_3000/1_2000/F_L_k_pB.npy")
    F_L_tot_tmp = np.load(f"../../omegaqeNBs/_old_kappa_F_L/_kappa_F_L_results_1perm_nob/{typs}/ACT/gmv/TEB/30_3000/1_2000/F_L_k.npy")
    F_L = InterpolatedUnivariateSpline(F_L_Ls, F_L_tmp)(Ls)
    F_L_tot = InterpolatedUnivariateSpline(F_L_Ls, F_L_tot_tmp)(Ls)
    NL1s = 1000
    main(typs, Ls, F_L, NL1s, F_L_tot)
