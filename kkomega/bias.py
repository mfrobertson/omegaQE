from fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from qe import QE
import vector
from cache.tools import getFileSep, path_exists

class Bias:

    class Holder:

        def __init__(self):
            self.sample_F_L_Ls = None
            self.F_L = None
            self.F_L_spline = None
            self.typs = None
            self.nu = None
            self.fields = None
            self.gmv = None
            self.C_inv = None
            self.Cl_kk = None
            self.Cov_kk = None
            self.Cl_gk = None
            self.Cl_Ik = None

        def get_Cl(self, typ):
            """

            Parameters
            ----------
            typ

            Returns
            -------

            """
            if typ == "kk":
                return self.Cl_kk
            elif typ == "gk":
                return self.Cl_gk
            elif typ == "Ik":
                return self.Cl_Ik
            else:
                raise ValueError(f"Type {typ} does not exist.")

        def get_Cov(self, typ):
            """

            Parameters
            ----------
            typ

            Returns
            -------

            """
            if typ == "kk":
                return self.Cov_kk
            elif typ == "gk":
                return self.Cl_gk
            elif typ == "Ik":
                return self.Cl_Ik
            else:
                raise ValueError(f"Type {typ} does not exist.")

    def __init__(self, N0_path, M_path=None, F_L_path=None, init_qe=True, exp="SO"):
        self.N0_path = N0_path
        self.F_L_path = F_L_path
        self.cache = self.Holder()
        self.exp = exp
        self.qe = QE(exp=self.exp, init=init_qe)
        N0_file = self._get_N0_file(gmv=True, fields="TEB")
        self.fisher = Fisher(N0_file, 2, True)
        if M_path is not None:
            self.fisher.setup_bispectra(M_path, 4000, 100)

    def _check_path(self, path):
        if not path_exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

    def _get_N0_file(self, gmv, fields):
        gmv_str = "gmv" if gmv else "single"
        sep = getFileSep()
        return self.N0_path + sep + self.exp + sep + gmv_str + sep + f"N0_{fields}_gradient.npy"

    def _reset_fisher_noise(self, gmv, fields):
        N0_file = self._get_N0_file(gmv, fields)
        self.fisher.reset_noise(N0_file, 2, True)

    def _build_F_L(self, typs, nu, fields, gmv):
        if self.F_L_path is not None:
            print("Using cached F_L")
            sep = getFileSep()
            gmv_str = "gmv_str" if gmv else "single"
            full_F_L_path = self.F_L_path+sep+self.exp+sep+gmv_str+sep+fields+sep+typs+sep+str(300)
            sample_Ls = np.load(full_F_L_path+sep+"ells.npy")
            F_L = np.load(full_F_L_path+sep+"F_L.npy")
            print("Full C_inv build...")
            self._reset_fisher_noise(gmv, fields)
            C_inv = self.fisher.get_C_inv(typs, Lmax=4000, nu=nu)
        else:
            print("Full F_L_build...")
            Ls1 = np.arange(3, 40, 2)
            Ls2 = np.logspace(1, 3, 100) * 4
            sample_Ls = np.concatenate((Ls1, Ls2))
            self._reset_fisher_noise(gmv, fields)
            _, F_L, C_inv = self.fisher.get_F_L(typs, Ls=sample_Ls, Ntheta=100, nu=nu, return_C_inv=True)
        self.cache.Cl_kk = self.fisher.get_Cl("kk", ellmax=4000, nu=nu)
        self.cache.Cov_kk = self.fisher.get_Cov("kk", ellmax=4000, nu=nu)
        self.cache.Cl_gk = self.fisher.get_Cl("gk", ellmax=4000, nu=nu)
        self.cache.Cl_Ik = self.fisher.get_Cl("Ik", ellmax=4000, nu=nu)
        self.cache.sample_F_L_Ls = sample_Ls
        self.cache.F_L = F_L
        self.cache.F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        self.cache.typs = typs
        self.cache.nu = nu
        self.cache.fields = fields
        self.cache.gmv = gmv
        self.cache.C_inv = C_inv

    def build_F_L(self, typs="kgI", fields="TEB", gmv=True, nu=353e9):
        self._build_F_L(typs, nu, fields, gmv)


    def _get_cov_invs(self, typ, typs, C_inv):
        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        return cov_inv1, cov_inv2


    def _mixed_bi_element(self, typ, typs, L1, L2, theta12, nu, C_inv):
        bi_typ = typ[:2] + "w"
        p = typ[2]
        q = typ[3]
        bi = self.fisher.bi.get_bispectrum(bi_typ, L1, L2, theta=theta12, M_spline=True, nu=nu)
        cov_inv1, cov_inv2 = self._get_cov_invs(typ, typs, C_inv)
        Ls = np.arange(np.size(cov_inv1))
        cov_inv1_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv1[1:])
        cov_inv2_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv2[1:])
        Cl_pk = self.cache.get_Cl(p+"k")
        Cl_qk = self.cache.get_Cl(q+"k")
        Cl_pk_spline = InterpolatedUnivariateSpline(Ls, Cl_pk)
        Cl_qk_spline = InterpolatedUnivariateSpline(Ls, Cl_qk)
        return bi * cov_inv1_spline(L1) * cov_inv2_spline(L2) * Cl_pk_spline(L1) * Cl_qk_spline(L2)


    def _mixed_bispectrum(self, typs, L1, L2, theta12, nu):
        L = self._get_third_L(L1, L2, theta12)
        F_L = self.cache.F_L_spline(L)
        typs = np.char.array(typs)
        C_inv = self.cache.C_inv
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        perms = 0
        mixed_bi = None
        for iii in np.arange(Ncombos):
            for jjj in np.arange(Ncombos):
                typ = combos[iii] + combos[jjj]
                mixed_bi_element = self._mixed_bi_element(typ, typs, L1, L2, theta12, nu, C_inv)
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

    def mixed_bispectrum(self, typs, L1, L2, theta12, nu=353e9, fields="TEB", gmv=True):
        """
        phi phi omega and phi phi kappa
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
            return 4*self.fisher.bi.get_bispectrum("kkw", L1, L2, theta=theta12, M_spline=True)/(L1**2 * L2**2)
        if typs == "conv":
            L = self._get_third_L(L1, L2, theta12)
            return 4*self.fisher.bi.get_bispectrum("kkk", L1, L2, L, M_spline=True)/(L1**2 * L2**2)
        if self.cache is None or self.cache.typs != typs or self.cache.nu != nu or self.cache.fields != fields or self.cache.gmv != gmv:
            self._build_F_L(typs, nu, fields, gmv)
        return self._mixed_bispectrum(list(typs), L1, L2, theta12, nu)

    def _get_third_L(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 + (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def bias_prep(self, bi_typ, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin):
        if bi_typ == "theory":
            if curl:
                bi_typ = "rot"
            else:
                bi_typ = "conv"
        samp1_1 = np.arange(Lmin, 40, 5)
        samp2_1 = np.logspace(1, 3, N_L1) * 4
        Ls1 = np.concatenate((samp1_1, samp2_1))
        Ls3 = np.linspace(Lmin, 4000, N_L3)
        dTheta1 = np.pi / Ntheta12
        thetas1 = np.linspace(0, np.pi-dTheta1, Ntheta12)
        dTheta3 = 2*np.pi / Ntheta13
        thetas3 = np.linspace(0, 2*np.pi-dTheta3, Ntheta13)
        return bi_typ, Ls1, thetas1, Ls3, thetas3, curl, Lmin

    def _bias_calc(self, XY, L, gmv, fields, bi_typ, Ls1, thetas1, Ls3, thetas3, curl, Lmin):
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
                w2 = 0 if (L2 < Lmin or L2 > 4000) else 1
                if w2 != 0:
                    bi = w2 * self.mixed_bispectrum(bi_typ, L1, L2, L1_vec.deltaphi(L2_vec), fields=fields, gmv=gmv)
                    L3_vec = vector.obj(rho=Ls3[None,:], phi=thetas3[:,None])
                    L5_vec = L1_vec - L3_vec
                    Ls5 = L5_vec.rho
                    w5 = np.ones(np.shape(Ls5))
                    w5[Ls5 < Lmin] = 0
                    w5[Ls5 > 4000] = 0
                    Xbar_Ybar = XY.replace("B", "E")
                    X_Ybar = XY[0] + XY[1].replace("B", "E")
                    Xbar_Y = XY[0].replace("B", "E") + XY[1]
                    C_L5 = self.qe.cmb[Xbar_Ybar].gradCl_spline(Ls5)
                    C_Ybar_L3 = self.qe.cmb[X_Ybar].gradCl_spline(Ls3[None,:])
                    C_Xbar_L3 = self.qe.cmb[Xbar_Y].gradCl_spline(Ls3[None,:])
                    L_A1_fac = (L5_vec @ L1_vec) * (L5_vec @ L2_vec)
                    L_C1_fac = (L3_vec @ L1_vec) * (L3_vec @ L2_vec)
                    g_XY = self.qe.weight_function(XY, L_vec, L3_vec, curl=curl, gmv=gmv, fields=fields)
                    g_YX = self.qe.weight_function(XY[::-1], L_vec, L3_vec, curl=curl, gmv=gmv, fields=fields)
                    L4_vec = L_vec - L3_vec
                    Ls4 = L4_vec.rho
                    w4 = np.ones(np.shape(Ls4))
                    w4[Ls4 < Lmin] = 0
                    w4[Ls4 > 4000] = 0
                    h_X_A1 = self.qe.geo_fac(XY[0], theta12=L5_vec.deltaphi(L4_vec))
                    h_Y_A1 = self.qe.geo_fac(XY[1], theta12=L5_vec.deltaphi(L3_vec))
                    h_X_C1 = self.qe.geo_fac(XY[0], theta12=L3_vec.deltaphi(L4_vec))
                    h_Y_C1 = self.qe.geo_fac(XY[1], theta12=L3_vec.deltaphi(L4_vec))
                    I_A1_theta1[jjj] = dTheta3 * dL3 * np.sum(bi * Ls3[None,:] * w4 * w5 * L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
                    I_C1_theta1[jjj] = dTheta3 * dL3 * np.sum(bi * Ls3[None,:] * w4 * L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))

            I_A1_L1[iii] = L1 * 2 * InterpolatedUnivariateSpline(thetas1, I_A1_theta1).integral(0, np.pi)
            I_C1_L1[iii] = L1 * 2 * InterpolatedUnivariateSpline(thetas1, I_C1_theta1).integral(0, np.pi)
        N_A1 = -0.5 * L**2 / ((2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(Lmin, 4000)
        N_C1 = 0.25 * L**2 / ((2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(Lmin, 4000)
        return N_A1, N_C1

    def _get_normalisation(self, fields, Ls, curl, gmv):
        self._reset_fisher_noise(gmv, fields)
        if curl:
            typ = "curl"
        else:
            typ = "phi"
        N0 = self.fisher.noise.get_N0(typ, ellmax=4000, ell_factors=False)
        sample_Ls = np.arange(np.size(N0))
        return InterpolatedUnivariateSpline(sample_Ls, N0)(Ls)

    def _bias(self, bi_typ, XY, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin):
        A = self._get_normalisation(fields=XY, Ls=Ls, curl=curl, gmv=False)
        N_Ls = np.size(Ls)
        if N_Ls == 1: Ls = np.ones(1) * Ls
        N_A1 = np.zeros(np.shape(Ls))
        N_C1 = np.zeros(np.shape(Ls))
        for iii, L in enumerate(Ls):
            N_A1_tmp, N_C1_tmp = self._bias_calc(XY, L, False, XY, *self.bias_prep(bi_typ, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin))
            N_A1[iii] = A[iii] * N_A1_tmp
            N_C1[iii] = A[iii] * N_C1_tmp
        return N_A1, N_C1

    def _bias_gmv(self, bi_typ, fields, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin):
        A = self._get_normalisation(fields=fields, Ls=Ls, curl=curl, gmv=True)
        N_Ls = np.size(Ls)
        if N_Ls == 1: Ls = np.ones(1) * Ls
        N_A1 = np.zeros(np.shape(Ls))
        N_C1 = np.zeros(np.shape(Ls))
        XYs = self.qe.parse_fields(fields)
        XYs = XYs[XYs != "BB"]
        for iii, L in enumerate(Ls):
            for XY in XYs:
                fac = 1 if XY[0] == XY[1] else 2
                N_A1_tmp, N_C1_tmp = self._bias_calc(XY, L, True, fields, *self.bias_prep(bi_typ, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin))
                N_A1[iii] += fac * A[iii] * N_A1_tmp
                N_C1[iii] += fac * A[iii] * N_C1_tmp
        return N_A1, N_C1


    def bias(self, bi_typ, fields, Ls, gmv=False, N_L1=30, N_L3=70, Ntheta12=25, Ntheta13=60, curl=True, Lmin=10):
        """

        Parameters
        ----------
        Ls
        N_L1
        N_L3
        Ntheta12
        Ntheta13
        curl

        Returns
        -------

        """
        if gmv:
            return self._bias_gmv(bi_typ, fields, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin)
        return self._bias(bi_typ, fields, Ls, N_L1, N_L3, Ntheta12, Ntheta13, curl, Lmin)
