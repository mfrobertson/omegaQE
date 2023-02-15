from fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from qe import QE
import vector
from cache.tools import getFileSep, path_exists

def _get_Cl(typ):
    if typ == "kk":
        return Cl_kk
    elif typ == "gk":
        return Cl_gk
    elif typ == "Ik":
        return Cl_Ik
    else:
        raise ValueError(f"Type {typ} does not exist.")


def _check_path(path):
    if not path_exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")


def _setup_noise(fish, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, data_dir=None):
    return fish.setup_noise(exp, qe, gmv, ps, L_cuts, iter, data_dir)


def _get_cached_F_L(F_L_path, typs):
    sep = getFileSep()
    fields = global_fish.qe
    exp = global_fish.exp
    gmv = global_fish.gmv
    gmv_str = "gmv" if gmv else "single"
    full_F_L_path = F_L_path+sep+typs+sep+exp+sep+gmv_str+sep+fields+sep+"30_3000"+sep+"1_2000"
    sample_Ls = np.load(full_F_L_path+sep+"Ls.npy")
    F_L = np.load(full_F_L_path+sep+"F_L.npy")
    return sample_Ls, F_L


def _get_cov_invs(typ, typs, C_inv):
    combo1_idx1 = np.where(typs == typ[0])[0][0]
    combo1_idx2 = np.where(typs == typ[2])[0][0]
    combo2_idx1 = np.where(typs == typ[1])[0][0]
    combo2_idx2 = np.where(typs == typ[3])[0][0]

    cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
    cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
    return cov_inv1, cov_inv2


def _bias_prep(bi_typ, fields, gmv, N_L1, N_L3, Ntheta12, Ntheta13, curl):
    if bi_typ == "theory":
        if curl:
            bi_typ = "rot"
        else:
            bi_typ = "conv"
    Lmin, Lmax = global_qe.get_Lmin_Lmax(fields, gmv, strict=False)
    Ls1 = global_qe.get_log_sample_Ls(Lmin, Lmax, N_L1, dL_small=2)
    Ls3 = np.linspace(Lmin, Lmax, N_L3)
    dTheta1 = np.pi / Ntheta12
    thetas1 = np.linspace(0, np.pi - dTheta1, Ntheta12)
    dTheta3 = 2 * np.pi / Ntheta13
    thetas3 = np.linspace(0, 2 * np.pi - dTheta3, Ntheta13)
    return bi_typ, Ls1, thetas1, Ls3, thetas3, curl


def _bias_calc(XY, L, gmv, fields, bi_typ, Ls1, thetas1, Ls3, thetas3, curl):
    Lmin, Lmax = global_qe.get_Lmin_Lmax(fields, gmv, strict=False)
    dTheta3 = thetas3[1] - thetas3[0]
    dL3 = Ls3[1] - Ls3[0]
    L_vec = vector.obj(rho=L, phi=0)
    I_A1_L1 = np.zeros(np.size(Ls1))
    I_C1_L1 = np.zeros(np.size(Ls1))
    for iii, L1 in enumerate(Ls1):
        I_A1_theta1 = np.zeros(np.size(thetas1))
        I_C1_theta1 = np.zeros(np.size(thetas1))
        for jjj, theta1 in enumerate(thetas1):
            L1_vec = vector.obj(rho=L1, phi=theta1)
            L2_vec = L_vec - L1_vec
            L2 = L2_vec.rho
            w2 = 0 if (L2 < Lmin or L2 > Lmax) else 1
            if w2 != 0:
                bi = w2 * mixed_bispectrum(bi_typ, L1, L2, L1_vec.deltaphi(L2_vec))
                L3_vec = vector.obj(rho=Ls3[None, :], phi=thetas3[:, None])
                L5_vec = L1_vec - L3_vec
                Ls5 = L5_vec.rho
                w5 = np.ones(np.shape(Ls5))
                w5[Ls5 < Lmin] = 0
                w5[Ls5 > Lmax] = 0
                Xbar_Ybar = XY.replace("B", "E")
                X_Ybar = XY[0] + XY[1].replace("B", "E")
                Xbar_Y = XY[0].replace("B", "E") + XY[1]
                C_L5 = global_qe.cmb[Xbar_Ybar].gradCl_spline(Ls5)
                C_Ybar_L3 = global_qe.cmb[X_Ybar].gradCl_spline(Ls3[None, :])
                C_Xbar_L3 = global_qe.cmb[Xbar_Y].gradCl_spline(Ls3[None, :])
                L_A1_fac = (L5_vec @ L1_vec) * (L5_vec @ L2_vec)
                L_C1_fac = (L3_vec @ L1_vec) * (L3_vec @ L2_vec)
                g_XY = global_qe.weight_function(XY, L_vec, L3_vec, curl=curl, gmv=gmv, fields=fields, apply_Lcuts=True)
                g_YX = global_qe.weight_function(XY[::-1], L_vec, L3_vec, curl=curl, gmv=gmv, fields=fields, apply_Lcuts=True)
                L4_vec = L_vec - L3_vec
                Ls4 = L4_vec.rho
                w4 = np.ones(np.shape(Ls4))
                w4[Ls4 < Lmin] = 0
                w4[Ls4 > Lmax] = 0
                h_X_A1 = global_qe.geo_fac(XY[0], theta12=L5_vec.deltaphi(L4_vec))
                h_Y_A1 = global_qe.geo_fac(XY[1], theta12=L5_vec.deltaphi(L3_vec))
                h_X_C1 = global_qe.geo_fac(XY[0], theta12=L3_vec.deltaphi(L4_vec))
                h_Y_C1 = global_qe.geo_fac(XY[1], theta12=L3_vec.deltaphi(L4_vec))
                I_A1_theta1[jjj] = dTheta3 * dL3 * np.sum(bi * Ls3[None, :] * w4 * w5 * L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
                I_C1_theta1[jjj] = dTheta3 * dL3 * np.sum(bi * Ls3[None, :] * w4 * L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))

        I_A1_L1[iii] = L1 * 2 * InterpolatedUnivariateSpline(thetas1, I_A1_theta1).integral(0, np.pi)
        I_C1_L1[iii] = L1 * 2 * InterpolatedUnivariateSpline(thetas1, I_C1_theta1).integral(0, np.pi)
    N_A1 = -0.5 * L ** 2 / ((2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(Lmin, Lmax)
    N_C1 = 0.25 * L ** 2 / ((2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(Lmin, Lmax)
    return N_A1, N_C1


def _get_normalisation(Ls, curl):
    if curl:
        typ = "curl"
    else:
        typ = "phi"
    N0 = global_fish.covariance.noise.get_N0(typ, ellmax=5000)
    sample_Ls = np.arange(np.size(N0))
    return InterpolatedUnivariateSpline(sample_Ls, N0)(Ls)


def _bias(bi_typ, fields, gmv, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl):
    A = _get_normalisation(Ls, curl)
    N_Ls = np.size(Ls)
    if N_Ls == 1: Ls = np.ones(1) * Ls
    N_A1 = np.zeros(np.shape(Ls))
    N_C1 = np.zeros(np.shape(Ls))
    XYs = global_qe.parse_fields(fields, unique=False) if gmv else [fields]
    for iii, L in enumerate(Ls):
        for XY in XYs:
            if gmv:
                N_A1_tmp, N_C1_tmp = _bias_calc(XY, L, True, fields, *_bias_prep(bi_typ, fields, True, N_L1, N_L3, Ntheta12, Ntheta13, curl))
            else:
                N_A1_tmp, N_C1_tmp = _bias_calc(XY, L, False, XY, *_bias_prep(bi_typ, XY, False, N_L1, N_L3, Ntheta12, Ntheta13, curl))
            N_A1[iii] += A[iii] * N_A1_tmp
            N_C1[iii] += A[iii] * N_C1_tmp
    return N_A1, N_C1


def _get_third_L(L1, L2, theta):
    return np.sqrt(L1 ** 2 + L2 ** 2 + (2 * L1 * L2 * np.cos(theta).astype("double"))).astype("double")


def _mixed_bi_innerloop(typ, typs, L1, L2, C_inv):
    p = typ[2]
    q = typ[3]
    cov_inv1, cov_inv2 = _get_cov_invs(typ, typs, C_inv)
    Ls = np.arange(np.size(cov_inv1))
    cov_inv1_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv1[1:])
    cov_inv2_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv2[1:])
    Cl_pk = _get_Cl(p + "k")
    Cl_qk = _get_Cl(q + "k")
    Cl_pk_spline = InterpolatedUnivariateSpline(Ls[1:], Cl_pk[1:])
    Cl_qk_spline = InterpolatedUnivariateSpline(Ls[1:], Cl_qk[1:])
    return cov_inv1_spline(L1) * cov_inv2_spline(L2) * Cl_pk_spline(L1) * Cl_qk_spline(L2)


def _mixed_bispectrum(typs, L1, L2, theta12, nu):
    L = _get_third_L(L1, L2, theta12)
    F_L = F_L_spline(L)
    typs = np.char.array(typs)
    all_combos = typs[:, None] + typs[None, :]
    combos = all_combos.flatten()
    Ncombos = np.size(combos)
    perms = 0
    mixed_bi = None
    for iii in np.arange(Ncombos):
        bi_ij = global_fish.bi.get_bispectrum(combos[iii]+"w", L1, L2, theta=theta12, M_spline=True, nu=nu)
        for jjj in np.arange(Ncombos):
            typ = combos[iii] + combos[jjj]
            mixed_bi_element = bi_ij * _mixed_bi_innerloop(typ, typs, L1, L2, C_inv)
            if combos[iii] != combos[jjj]:
                factor = 1
            else:
                factor = 1
            perms += factor
            if mixed_bi is None:
                mixed_bi = mixed_bi_element
            else:
                mixed_bi += factor * mixed_bi_element
    if perms != np.size(typs) ** 4:
        raise ValueError(f"{perms} permutations computed, should be {np.size(typs) ** 4}")
    return 4 / (F_L * L1 ** 2 * L2 ** 2) * mixed_bi


def mixed_bispectrum(typs, L1, L2, theta12, nu=353e9):
    """

    Parameters
    ----------
    typs
    L1
    L2
    theta12
    nu

    Returns
    -------

    """
    if typs == "rot":
        return 4 * global_fish.bi.get_bispectrum("kkw", L1, L2, theta=theta12, M_spline=True) / (L1 ** 2 * L2 ** 2)
    if typs == "conv":
        L = _get_third_L(L1, L2, theta12)
        return 4 * global_fish.bi.get_bispectrum("kkk", L1, L2, L, M_spline=True) / (L1 ** 2 * L2 ** 2)
    return _mixed_bispectrum(list(typs), L1, L2, theta12, nu)


def bias(Ls, bi_typ, curl, exp=None, qe_fields=None, gmv=None, ps=None, L_cuts=None, iter=None, data_dir=None, F_L_path="_results/F_L_results", qe_setup_path=None, N_L1=30, N_L3=70, Ntheta12=25, Ntheta13=60):
    """

    Parameters
    ----------
    Ls
    bi_typ
    curl
    exp
    qe_fields
    gmv
    ps
    L_cuts
    iter
    data_dir
    F_L_path

    Returns
    -------

    """
    global global_qe, global_fish
    global_fish = Fisher()
    global_fish.setup_noise(exp, qe_fields, gmv, ps, L_cuts, iter, data_dir)
    global_fish.setup_bispectra()
    if qe_setup_path is None:
        global_qe = QE(exp=global_fish.exp, init=True, L_cuts=global_fish.L_cuts)
    else:
        global_qe = QE(exp=global_fish.exp, init=False, L_cuts=global_fish.L_cuts)
        parsed_fields_all = global_qe.parse_fields(includeBB=True)
        Cls = np.load(qe_setup_path)
        for iii, field in enumerate(parsed_fields_all):
            lenCl = Cls[iii, 0, :]
            gradCl = Cls[iii, 1, :]
            N = Cls[iii, 2, :]
            global_qe.initialise_manual(field, lenCl, gradCl, N)
        global_qe.initialise()


    if bi_typ != "theory":
        global F_L_spline, C_inv, Cl_kk, Cl_gk, Cl_Ik
        sample_Ls, F_L = _get_cached_F_L(F_L_path, bi_typ)
        F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        C_inv = global_fish.covariance.get_C_inv(bi_typ, Lmax=int(np.ceil(np.max(sample_Ls))), nu=353e9)

        Cl_kk = global_fish.covariance.get_Cl("kk", ellmax=5000)
        Cl_gk = global_fish.covariance.get_Cl("gk", ellmax=5000)
        Cl_Ik = global_fish.covariance.get_Cl("Ik", ellmax=5000)

    Ls = np.ones(1, dtype=int)*Ls if np.size(Ls) == 1 else Ls
    return _bias(bi_typ, global_fish.qe, global_fish.gmv, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl)
