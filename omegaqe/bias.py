import omegaqe
from omegaqe.fisher import Fisher
from omegaqe.qe import QE
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import vector
from omegaqe.tools import getFileSep, path_exists

def _get_Cl_spline(typ):
    if typ == "kk":
        return Cl_kk_spline
    elif typ == "gk":
        return Cl_gk_spline
    elif typ == "Ik":
        return Cl_Ik_spline
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


def _get_cov_inv_spline(typ, typs):
    combo1_idx1 = np.where(typs == typ[0])[0][0]
    combo1_idx2 = np.where(typs == typ[2])[0][0]
    combo2_idx1 = np.where(typs == typ[1])[0][0]
    combo2_idx2 = np.where(typs == typ[3])[0][0]

    cov_inv1 = C_inv_splines[combo1_idx1][combo1_idx2]
    cov_inv2 = C_inv_splines[combo2_idx1][combo2_idx2]
    return cov_inv1, cov_inv2

def _get_N2_innerloop(XY, curl, gmv, fields, dThetal, dl, bi, w_prim, w_primprim, ls, l_primprims, L_vec, L1_vec, L2_vec, l_vec, l_prim_vec, l_primprim_vec, noise):
    Xbar_Ybar = XY.replace("B", "E")
    X_Ybar = XY[0] + XY[1].replace("B", "E")
    Xbar_Y = XY[0].replace("B", "E") + XY[1]
    C_l_primprim = global_qe.cmb[Xbar_Ybar].gradCl_spline(l_primprims)
    C_Ybar_l = global_qe.cmb[X_Ybar].gradCl_spline(ls[None, :])
    C_Xbar_l = global_qe.cmb[Xbar_Y].gradCl_spline(ls[None, :])
    L_A1_fac = (l_primprim_vec @ L1_vec) * (l_primprim_vec @ L2_vec)
    L_C1_fac = (l_vec @ L1_vec) * (l_vec @ L2_vec)
    g_XY = global_qe.weight_function(XY, L_vec, l_vec, curl=curl, gmv=gmv, fields=fields, apply_Lcuts=True)
    h_X_A1 = global_qe.geo_fac(XY[0], theta12=l_primprim_vec.deltaphi(l_vec))   # this is different to N32 paper
    h_Y_A1 = global_qe.geo_fac(XY[1], theta12=l_primprim_vec.deltaphi(l_prim_vec))       # ^^^ same ^^^
    h_X_C1 = global_qe.geo_fac(XY[0], theta12=l_vec.deltaphi(l_prim_vec))
    h_Y_C1 = global_qe.geo_fac(XY[1], theta12=l_vec.deltaphi(l_prim_vec))
    I_A1_theta1 = (-1) * dThetal * dl * np.sum(bi * ls[None, :] * w_prim * w_primprim * L_A1_fac * C_l_primprim * g_XY * h_X_A1 * h_Y_A1)
    C1_fac = 1 if gmv else 0.5
    I_C1_theta1 = C1_fac * dThetal * dl * np.sum(bi * ls[None, :] * w_prim * L_C1_fac * C_Ybar_l * g_XY * h_Y_C1)
    if not gmv:
        g_YX = global_qe.weight_function(XY[::-1], L_vec, l_vec, curl=curl, gmv=gmv, fields=fields, apply_Lcuts=True)
        I_C1_theta1 += dThetal * dl * np.sum(bi * ls[None, :] * w_prim * L_C1_fac * (C_Xbar_l * g_YX * h_X_C1))
    return I_A1_theta1 + I_C1_theta1

def _get_N1_innerloop(XY, curl, gmv, fields, dThetal, dl, alpha, w_prim, w_primprim, ls, l_primprims, L_vec, L1_vec, L2_vec, l_vec, l_prim_vec, l_primprim_vec, noise):
    g_XY = global_qe.weight_function(XY, L_vec, l_vec, curl=True, gmv=gmv, fields=fields, apply_Lcuts=True)
    XprimsYprims = global_qe.parse_fields(fields, unique=False) if gmv else [fields]
    inner_sum = np.zeros(np.shape(w_prim))
    for XprimYprim in XprimsYprims:
        XXprim = XY[0]+XprimYprim[0]
        YYprim = XY[1] + XprimYprim[1]
        C_XXprim_l = global_qe.cmb[XXprim].lenCl_spline(ls)
        if noise:
            C_XXprim_l += global_qe.cmb[XXprim].N_spline(ls)
        g_XprimYprim = global_qe.weight_function(XprimYprim, L1_vec, l_vec, curl=False, gmv=gmv, fields=fields, apply_Lcuts=True)
        resp_YYprim = global_qe.response(YYprim, L2_vec, l_prim_vec, curl=False)
        inner_sum += g_XprimYprim * C_XXprim_l * resp_YYprim
    return 4 * dThetal * dl * np.sum(alpha * g_XY * inner_sum * ls[None, :] * w_prim)

def _get_N0_innerloop(XY, curl, gmv, fields, dThetal, dl, beta, w_prim, w_primprim, ls, l_primprims, L_vec, L1_vec, L2_vec, l_vec, l_prim_vec, l_primprim_vec, noise):
    g_XY = global_qe.weight_function(XY, L_vec, l_vec, curl=True, gmv=gmv, fields=fields, apply_Lcuts=True)
    XpYps = global_qe.parse_fields(fields, unique=False) if gmv else [fields]
    inner_sum = np.zeros(np.shape(w_prim))
    for XpYp in XpYps:
        g_XpYp = global_qe.weight_function(XpYp, L1_vec, l_vec, curl=False, gmv=gmv, fields=fields, apply_Lcuts=True)
        XXp = XY[0] + XpYp[0]
        C_XXp_l = global_qe.cmb[XXp].lenCl_spline(ls)
        if noise:
            C_XXp_l += global_qe.cmb[XXp].N_spline(ls)
        for XppYpp in XpYps:
            g_XppYpp= global_qe.weight_function(XppYpp, L2_vec, l_prim_vec, curl=False, gmv=gmv, fields=fields, apply_Lcuts=True)
            YXpp = XY[1] + XppYpp[0]
            YpYpp = XpYp[1] + XppYpp[1]
            C_YXpp_l = global_qe.cmb[YXpp].lenCl_spline(l_prim_vec.rho)
            if noise:
                C_YXpp_l += global_qe.cmb[YXpp].N_spline(l_prim_vec.rho)
            C_YpYpp_l = global_qe.cmb[YpYpp].lenCl_spline(l_primprims)
            if noise:
                C_YpYpp_l += global_qe.cmb[YpYpp].N_spline(l_primprims)
            inner_sum += g_XpYp * g_XppYpp * C_XXp_l * C_YXpp_l * C_YpYpp_l
    return 4 * dThetal * dl * np.sum(beta * g_XY * inner_sum * ls[None, :] * w_prim * w_primprim)

def _bias_calc(bias_typ, XY, L, gmv, fields, bi_typ, curl, L1s, thetas1, ls, thetasl, verbose, noise):

    if bias_typ == "N2" or bias_typ == "N32":
        innerloop_func = _get_N2_innerloop
    elif bias_typ == "N1":
        innerloop_func = _get_N1_innerloop
    elif bias_typ == "N0":
        innerloop_func = _get_N0_innerloop
    else:
        raise ValueError(f"Bias type {bias_typ} does not match built types.")
    Lmin, Lmax = global_qe.get_Lmin_Lmax(fields, gmv, strict=False)
    dThetal = thetasl[1] - thetasl[0]
    dl = ls[1] - ls[0]
    L_vec = vector.obj(rho=L, phi=0)
    I_L1 = np.zeros(np.size(L1s))
    for iii, L1 in enumerate(L1s):
        if verbose: print(f"    L1 = {L1} ({iii}/{np.size(L1s) - 1})")
        I_theta1 = np.zeros(np.size(thetas1))
        for jjj, theta1 in enumerate(thetas1):
            L1_vec = vector.obj(rho=L1, phi=theta1)
            L2_vec = L_vec - L1_vec
            L2 = L2_vec.rho
            w2 = 0 if (L2 < Lmin or L2 > Lmax) else 1
            if w2 != 0:
                bi = w2 * mixed_bispectrum(bias_typ, bi_typ, L1, L2, L1_vec.deltaphi(L2_vec))
                l_vec = vector.obj(rho=ls[None, :], phi=thetasl[:, None])
                l_primprim_vec = L1_vec - l_vec
                l_primprims = l_primprim_vec.rho
                w_primprim = np.ones(np.shape(l_primprims))
                w_primprim[l_primprims < Lmin] = 0
                w_primprim[l_primprims > Lmax] = 0
                l_prim_vec = L_vec - l_vec
                l_prims = l_prim_vec.rho
                w_prim = np.ones(np.shape(l_prims))
                w_prim[l_prims < Lmin] = 0
                w_prim[l_prims > Lmax] = 0
                if np.sum(w_prim) != 0 and np.sum(w_primprim) != 0:
                    I_theta1[jjj] = innerloop_func(XY, curl, gmv, fields, dThetal, dl, bi, w_prim, w_primprim, ls, l_primprims, L_vec, L1_vec, L2_vec, l_vec, l_prim_vec, l_primprim_vec, noise)

        I_L1[iii] = L1 * 2 * InterpolatedUnivariateSpline(thetas1, I_theta1).integral(0, np.pi)
    N = InterpolatedUnivariateSpline(L1s, I_L1).integral(Lmin, Lmax) / ((2 * np.pi) ** 4)
    return N


def _get_normalisation(Ls, curl):
    if curl:
        return Ls**2/2 / N0_w_spline(Ls)
    else:
        return Ls**2/2 / N0_k_spline(Ls)

def _setup_norms():
    N0_w = global_fish.covariance.noise.get_N0("omega", ellmax=5000)
    sample_Ls_w = np.arange(np.size(N0_w))
    N0_k = global_fish.covariance.noise.get_N0("kappa", ellmax=5000)
    sample_Ls_k = np.arange(np.size(N0_w))
    return InterpolatedUnivariateSpline(sample_Ls_w, N0_w), InterpolatedUnivariateSpline(sample_Ls_k, N0_k)

def _bias_prep(fields, gmv, N_L1, N_l, Ntheta12, Ntheta1l):
    Lmin, Lmax = global_qe.get_Lmin_Lmax(fields, gmv, strict=False)
    L1s = global_qe.get_log_sample_Ls(Lmin, Lmax, N_L1, dL_small=2)  # dL_small should be 2 if not enough L steps (which is default)
    ls = np.linspace(Lmin, Lmax, N_l)
    dTheta1 = np.pi / Ntheta12
    thetas1 = np.linspace(0, np.pi - dTheta1, Ntheta12)
    dThetal = 2 * np.pi / Ntheta1l
    thetasl = np.linspace(0, 2 * np.pi - dThetal, Ntheta1l)
    return L1s, thetas1, ls, thetasl

def _bias(bias_typ, bi_typ, fields, gmv, Ls, N_L1, N_L3, Ntheta12, Ntheta13, verbose, noise):
    curl = False if bias_typ == "N32" else True
    A = _get_normalisation(Ls, curl)
    N_Ls = np.size(Ls)
    if N_Ls == 1: Ls = np.ones(1) * Ls
    N = np.zeros(np.shape(Ls))
    XYs = global_qe.parse_fields(fields, unique=False) if gmv else [fields]
    for iii, L in enumerate(Ls):
        if verbose: print(f"L = {L} ({iii}/{N_Ls-1})")
        for XY in XYs:
            if verbose: print(f"  XY = {XY}")
            if gmv:
                N_tmp = _bias_calc(bias_typ, XY, L, True, fields, bi_typ, curl, *_bias_prep(fields, True, N_L1, N_L3, Ntheta12, Ntheta13), verbose=verbose, noise=noise)
            else:
                N_tmp = _bias_calc(bias_typ, XY, L, False, XY, bias_typ, curl, *_bias_prep(XY, False, N_L1, N_L3, Ntheta12, Ntheta13), verbose=verbose, noise=noise)
            N[iii] += N_tmp / A[iii]
    return N


def _get_third_L(L1, L2, theta):
    return np.sqrt(L1 ** 2 + L2 ** 2 + (2 * L1 * L2 * np.cos(theta).astype("double"))).astype("double")


def _mixed_bi_innerloop(typ, typs, L1, L2):
    p = typ[2]
    q = typ[3]
    cov_inv1_spline, cov_inv2_spline = _get_cov_inv_spline(typ, typs)
    Cl_pk_spline = _get_Cl_spline(p + "k")
    Cl_qk_spline = _get_Cl_spline(q + "k")
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
            mixed_bi_element = bi_ij * _mixed_bi_innerloop(typ, typs, L1, L2)
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

def _alpha(typs, L1, L2, theta12, nu):
    L = _get_third_L(L1, L2, theta12)
    F_L = F_L_spline(L)
    typs = np.char.array(typs)
    all_combos = typs[:, None] + typs[None, :]
    combos = all_combos.flatten()
    Ncombos = np.size(combos)
    mixed_bi = None
    for iii in np.arange(Ncombos):
        bi_ij = global_fish.bi.get_bispectrum(combos[iii] + "w", L1, L2, theta=theta12, M_spline=True, nu=nu)
        for q in typs:
            if q != "k":
                kq = "k" + q
                cov_inv1_spline, cov_inv2_spline = _get_cov_inv_spline(combos[iii]+kq, typs)
                Cl_qk_spline = _get_Cl_spline(q + "k")
                mixed_bi_element = bi_ij * cov_inv1_spline(L1) * cov_inv2_spline(L2) * Cl_qk_spline(L2)
                if mixed_bi is None:
                    mixed_bi = mixed_bi_element
                else:
                    mixed_bi += mixed_bi_element
    A_k = _get_normalisation(curl=False, Ls=L1)
    return mixed_bi / A_k * 2/(L2**2 * F_L)

def _beta(typs, L1, L2, theta12, nu):
    L = _get_third_L(L1, L2, theta12)
    F_L = F_L_spline(L)
    typs = np.char.array(typs)
    all_combos = typs[:, None] + typs[None, :]
    combos = all_combos.flatten()
    Ncombos = np.size(combos)
    mixed_bi = None
    for iii in np.arange(Ncombos):
        bi_ij = global_fish.bi.get_bispectrum(combos[iii] + "w", L1, L2, theta=theta12, M_spline=True, nu=nu)
        cov_inv1_spline, cov_inv2_spline = _get_cov_inv_spline(combos[iii] + "kk", typs)
        mixed_bi_element = bi_ij * cov_inv1_spline(L1) * cov_inv2_spline(L2)
        if mixed_bi is None:
            mixed_bi = mixed_bi_element
        else:
            mixed_bi += mixed_bi_element
    A_k_L1 = _get_normalisation(curl=False, Ls=L1)
    A_k_L2 = _get_normalisation(curl=False, Ls=L2)
    return mixed_bi / A_k_L1 / A_k_L2 / F_L

def mixed_bispectrum(bias_typ, bi_typ, L1, L2, theta12, nu=353e9):
    """

    Parameters
    ----------
    bias_typ
    bi_typ
    L1
    L2
    theta12
    nu

    Returns
    -------

    """

    if bias_typ == "N2":
        if bi_typ == "theory":
            return 4 * global_fish.bi.get_bispectrum("kkw", L1, L2, theta=theta12, M_spline=True) / (L1 ** 2 * L2 ** 2)
        return _mixed_bispectrum(list(bi_typ), L1, L2, theta12, nu)
    if bias_typ == "N32":
        L = _get_third_L(L1, L2, theta12)
        return 4 * global_fish.bi.get_bispectrum("kkk", L1, L2, L, M_spline=True) / (L1 ** 2 * L2 ** 2)
    if bias_typ == "N1":
        if bi_typ == "theory":
            return 2 * global_fish.bi.get_bispectrum("kkw", L1, L2, theta=theta12, M_spline=True) / (L1 ** 2)
        return _alpha(list(bi_typ), L1, L2, theta12, nu)
    if bias_typ == "N0":
        if bi_typ == "theory":
            return global_fish.bi.get_bispectrum("kkw", L1, L2, theta=theta12, M_spline=True)
        return _beta(list(bi_typ), L1, L2, theta12, nu)

def _build_C_inv_splines(C_inv, bi_typ):
    N_fields = np.size(list(bi_typ))
    C_inv_splines = np.empty((N_fields, N_fields), dtype=InterpolatedUnivariateSpline)
    Ls = np.arange(np.size(C_inv[0][0]))
    for iii in range(N_fields):
        for jjj in range(N_fields):
            C_inv_ij = C_inv[iii, jjj]
            C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[1:], C_inv_ij[1:])
    return C_inv_splines


def bias(bias_typ, Ls, bi_typ="theory", exp=None, qe_fields=None, gmv=None, ps=None, L_cuts=None, iter=None, data_dir=None, F_L_path=f"{omegaqe.RESULTS_DIR}{getFileSep()}F_L_results", qe_setup_path=None, N_L1=30, N_L3=70, Ntheta12=25, Ntheta13=60, verbose=False, noise=True):

    global global_qe, global_fish, N0_w_spline, N0_k_spline
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
    N0_w_spline, N0_k_spline = _setup_norms()


    if bi_typ != "theory":
        if verbose: print("Caching bispectrum splines")
        global F_L_spline, C_inv_splines, Cl_kk_spline, Cl_gk_spline, Cl_Ik_spline
        sample_Ls, F_L = _get_cached_F_L(F_L_path, bi_typ)
        F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        C_inv = global_fish.covariance.get_C_inv(bi_typ, Lmax=int(np.ceil(np.max(sample_Ls))), nu=353e9)
        C_inv_splines = _build_C_inv_splines(C_inv, bi_typ)

        Cl_kk = global_fish.covariance.get_Cl("kk", ellmax=5000)
        Ls_sample = np.arange(np.size(Cl_kk))
        Cl_kk_spline = InterpolatedUnivariateSpline(Ls_sample[1:], Cl_kk[1:])
        Cl_gk = global_fish.covariance.get_Cl("gk", ellmax=5000)
        Cl_gk_spline = InterpolatedUnivariateSpline(Ls_sample[1:], Cl_gk[1:])
        Cl_Ik = global_fish.covariance.get_Cl("Ik", ellmax=5000)
        Cl_Ik_spline = InterpolatedUnivariateSpline(Ls_sample[1:], Cl_Ik[1:])

    Ls = np.ones(1, dtype=int)*Ls if np.size(Ls) == 1 else Ls
    return _bias(bias_typ, bi_typ, global_fish.qe, global_fish.gmv, Ls, N_L1, N_L3, Ntheta12, Ntheta13, verbose, noise)
