from fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmology import Cosmology
from qe import QE
import vector


class Bias:

    class Holder:

        def __init__(self):
            self.F_L_spline = None
            self.typs = None
            self.nu = None
            self.C_inv = None
            self.Cl_kk = None
            self.Cov_kk = None
            self.Cl_gk = None
            self.Cl_Ik = None
            self.Cl_TT_spline = None
            self.Cl_gradT_spline = None
            self.N_T_spline = None

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


    def __init__(self, N0_file, N0_offset=0, N0_ell_factors=True, M_path=None):
        self.fisher = Fisher(N0_file, N0_offset, N0_ell_factors)
        self.N0_offset = N0_offset
        self.N0_ell_factors = N0_ell_factors
        self._cache = self.Holder()
        self._cosmo = Cosmology()
        self._qe = QE()
        if M_path is not None:
            self.fisher.setup_bispectra(M_path, 4000, 100)


    def _build_F_L(self, typs, nu):
        print("F_L_build")
        Ls1 = np.arange(30, 40, 2)
        Ls2 = np.logspace(1, 3, 100) * 4
        sample_Ls = np.concatenate((Ls1, Ls2))
        _, F_L, C_inv = self.fisher.get_F_L(typs, Ls=sample_Ls, Ntheta=100, nu=nu, return_C_inv=True)
        self._cache.Cl_kk = self.fisher.get_Cl("kk", ellmax=4000, nu=nu)
        self._cache.Cov_kk = self.fisher.get_Cov("kk", ellmax=4000, nu=nu)
        self._cache.Cl_gk = self.fisher.get_Cl("gk", ellmax=4000, nu=nu)
        self._cache.Cl_Ik = self.fisher.get_Cl("Ik", ellmax=4000, nu=nu)
        self._cache.F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        self._cache.typs = typs
        self._cache.nu = nu
        self._cache.C_inv = C_inv

    def _get_cov_invs(self, typ, typs, C_inv):
        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        return cov_inv1, cov_inv2

    # def _mixed_bi_element(self, typ, typs, L, L1, L2, nu, C_inv):
    #     bi_typ = typ[:2] + "w"
    #     p = typ[2]
    #     q = typ[3]
    #     bi = self.fisher.bi.get_bispectrum(bi_typ, L1, L2, L, M_spline=True, nu=nu)
    #     cov_inv1, cov_inv2 = self._get_cov_invs(typ, typs, C_inv)
    #     Ls = np.arange(np.size(cov_inv1))
    #     cov_inv1_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv1[1:])
    #     cov_inv2_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv2[1:])
    #     Cl_pk = self._cache.get_Cl(p+"k")
    #     Cl_qk = self._cache.get_Cl(q+"k")
    #     Cl_pk_spline = InterpolatedUnivariateSpline(Ls, Cl_pk)
    #     Cl_qk_spline = InterpolatedUnivariateSpline(Ls, Cl_qk)
    #     return bi * cov_inv1_spline(L1) * cov_inv2_spline(L2) * Cl_pk_spline(L1) * Cl_qk_spline(L2)
    #
    #
    # def _mixed_bispectrum(self, typs, L, L1, L2, nu):
    #     F_L = self._cache.F_L_spline(L)
    #     typs = np.char.array(typs)
    #     C_inv = self._cache.C_inv
    #     all_combos = typs[:, None] + typs[None, :]
    #     combos = all_combos.flatten()
    #     Ncombos = np.size(combos)
    #     perms = 0
    #     mixed_bi = None
    #     for iii in np.arange(Ncombos):
    #         for jjj in np.arange(Ncombos):
    #             typ = combos[iii] + combos[jjj]
    #             mixed_bi_element = self._mixed_bi_element(typ, typs, L, L1, L2, nu, C_inv)
    #             if combos[iii] != combos[jjj]:
    #                 factor = 1
    #             else:
    #                 factor = 1
    #             perms += factor
    #             if mixed_bi is None:
    #                 mixed_bi = mixed_bi_element
    #             else:
    #                 mixed_bi += factor * mixed_bi_element
    #     if perms != np.size(typs) ** 4:
    #         raise ValueError(f"{perms} permutations computed, should be {np.size(typs) ** 4}")
    #     return 4 / (F_L * L1 ** 2 * L2 ** 2) * mixed_bi
    #
    # def mixed_bispectrum(self, typs, L, L1, L2, nu=353e9):
    #     """
    #
    #     Parameters
    #     ----------
    #     typs
    #     L
    #     L1
    #     L2
    #     nu
    #
    #     Returns
    #     -------
    #
    #     """
    #     if typs == "theory":
    #         return 4*self.fisher.bi.get_bispectrum("kkw", L1, L2, L)/(L1**2 * L2**2)
    #     if self._cache is None or self._cache.typs != typs or self._cache.nu != nu:
    #         self._build_F_L(typs, nu)
    #     return self._mixed_bispectrum(list(typs), L, L1, L2, nu)


    def _mixed_bi_element(self, typ, typs, L1, L2, theta12, nu, C_inv):
        bi_typ = typ[:2] + "w"
        p = typ[2]
        q = typ[3]
        bi = self.fisher.bi.get_bispectrum(bi_typ, L1, L2, theta=theta12, M_spline=True, nu=nu)
        cov_inv1, cov_inv2 = self._get_cov_invs(typ, typs, C_inv)
        Ls = np.arange(np.size(cov_inv1))
        cov_inv1_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv1[1:])
        cov_inv2_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv2[1:])
        Cl_pk = self._cache.get_Cl(p+"k")
        Cl_qk = self._cache.get_Cl(q+"k")
        Cl_pk_spline = InterpolatedUnivariateSpline(Ls, Cl_pk)
        Cl_qk_spline = InterpolatedUnivariateSpline(Ls, Cl_qk)
        return bi * cov_inv1_spline(L1) * cov_inv2_spline(L2) * Cl_pk_spline(L1) * Cl_qk_spline(L2)


    def _mixed_bispectrum(self, typs, L1, L2, theta12, nu):
        L = self._get_third_L(L1, L2, theta12)
        F_L = self._cache.F_L_spline(L)
        typs = np.char.array(typs)
        C_inv = self._cache.C_inv
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

    def mixed_bispectrum(self, typs, L1, L2, theta12, nu=353e9):
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
        if typs == "theory":
            return 4*self.fisher.bi.get_bispectrum("kkw", L1, L2, theta=theta12)/(L1**2 * L2**2)
        if typs == "conv":
            L = self._get_third_L(L1, L2, theta12)
            return 4*self.fisher.bi.get_bispectrum("kkk", L1, L2, L)/(L1**2 * L2**2)
        if self._cache is None or self._cache.typs != typs or self._cache.nu != nu:
            self._build_F_L(typs, nu)
        return self._mixed_bispectrum(list(typs), L1, L2, theta12, nu)

    def _get_third_L(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    # def bias(self, XY, Ls, N_L1=20, N_L3=20, Ntheta12=50, Ntheta13=10):
    #     A = self._qe.normalisation(XY, Ls, dL=2)
    #     print(f"A: {A}")
    #     samp1_1 = np.arange(30, 40, 5)
    #     samp2_1 = np.logspace(1, 3, N_L1) * 4
    #     Ls1 = np.concatenate((samp1_1, samp2_1))
    #     samp1_2 = np.arange(30, 40, 5)
    #     samp2_2 = np.logspace(1, 3, N_L3) * 4
    #     Ls3 = np.concatenate((samp1_2, samp2_2))
    #     dTheta12 = np.pi / Ntheta12
    #     thetas12 = np.arange(dTheta12, np.pi + dTheta12, dTheta12, dtype=float)
    #     dtheta13 = np.pi / Ntheta13
    #     thetas13 = np.arange(dtheta13, np.pi + dtheta13, dtheta13, dtype=float)
    #     N_Ls = np.size(Ls)
    #     if N_Ls == 1:
    #         Ls = np.ones(1) * Ls
    #     A1 = np.zeros(np.shape(Ls))
    #     C1 = np.zeros(np.shape(Ls))
    #     for lll, L in enumerate(Ls):
    #         print(L)
    #         I_A1_L1 = np.zeros(np.size(Ls1))
    #         I_C1_L1 = np.zeros(np.size(Ls1))
    #         for iii, L1 in enumerate(Ls1):
    #             for theta12 in thetas12:
    #                 theta_L2 = np.arcsin(L1 * np.sin(theta12) / L)  # Unsure
    #                 theta_1L = theta12 - theta_L2
    #                 # L2 = self._get_third_L(L1, L, theta_1L)
    #                 L2 = np.abs(L*np.sin(theta_1L)/np.sin(theta12))
    #                 # print(theta_1L)
    #                 # print(np.pi-theta12)
    #                 # print(L2)
    #                 # print(L*np.sin(theta_1L)/np.sin(np.pi-theta12))
    #                 # print("------")
    #                 w2 = 0 if (L2 < 3 or np.isnan(L2) or L2 > 6000) else 1
    #                 bi = w2 * 4 * self.mixed_bispectrum("theory", L1, L2, theta12) / (L1 ** 2 * L2 ** 2)
    #                 I_A1_L3 = np.zeros(np.size(Ls3))
    #                 I_C1_L3 = np.zeros(np.size(Ls3))
    #                 if w2 != 0:
    #                     for jjj, L3 in enumerate(Ls3):
    #                         Ls5 = self._get_third_L(L1, L3, thetas13)
    #                         w5 = np.ones(np.shape(Ls5))
    #                         w5[Ls5 < 3] = 0
    #                         w5[Ls5 > 6000] = 0
    #                         Xbar_Ybar = XY.replace("B", "E")
    #                         X_Ybar = XY[0] + XY[1].replace("B","E")
    #                         Xbar_Y = XY[0].replace("B","E") + XY[1]
    #                         C_L5 = w5 * self._qe.cmb[Xbar_Ybar].Cl_spline(Ls5)
    #                         C_Ybar_L3 = w5 * self._qe.cmb[X_Ybar].Cl_spline(L3)
    #                         C_Xbar_L3 = w5 * self._qe.cmb[Xbar_Y].Cl_spline(L3)
    #                         L1_L1 = L1 ** 2
    #                         L1_L3 = L1 * L3 * np.cos(thetas13)
    #                         L1_L2 = L1 * L2 * np.cos(theta12)
    #                         L2_L3 = L2 * L3 * np.cos(theta12 - thetas13)
    #                         L_A1_fac = (L1_L1 - L1_L3) * (L1_L2 - L2_L3)
    #                         L_C1_fac = L1_L3 * L2_L3
    #                         thetas_L3 = thetas13 - theta_1L
    #                         g_XY = self._qe.weight_function(XY, L3, L, thetas_L3, curl=True)
    #                         g_XY[np.isnan(g_XY)] = 0
    #                         g_YX = self._qe.weight_function(XY[::-1], L3, L, thetas_L3, curl=True)
    #                         g_YX[np.isnan(g_YX)] = 0
    #                         # print(f"thetas_L3: {thetas_L3}")
    #                         # print(f"g_XY: {g_XY}")
    #                         # print(f"g_YX: {g_YX}")
    #
    #                         thetas_51 = np.arcsin(L3*np.sin(thetas13)/Ls5)
    #                         thetas_53 = thetas_51 + thetas13
    #
    #                         Ls4 = self._get_third_L(L3, L, thetas_L3)
    #                         # print(f"Ls4: {Ls4}")
    #                         thetas_4L = np.arcsin(L3*np.sin(thetas_L3)/Ls4)
    #                         # print(f"thetas_4L: {thetas_4L}")
    #                         thetas_34 = 2*np.pi - thetas_4L - thetas_L3
    #                         # print(f"thetas_34: {thetas_34}")
    #                         thetas_54 = thetas_34 + thetas_53
    #                         # print(f"thetas_54: {thetas_54}")
    #
    #                         h_X_A1 = self._qe.geo_fac(XY[0], theta12=thetas_54)
    #                         h_X_A1[np.isnan(h_X_A1)] = 0
    #                         h_Y_A1 = self._qe.geo_fac(XY[1], theta12=thetas_53)
    #                         h_Y_A1[np.isnan(h_Y_A1)] = 0
    #                         h_X_C1 = self._qe.geo_fac(XY[0], theta12=thetas_34)
    #                         h_X_C1[np.isnan(h_X_C1)] = 0
    #                         h_Y_C1 = self._qe.geo_fac(XY[1], theta12=thetas_34)
    #                         h_Y_C1[np.isnan(h_Y_C1)] = 0
    #                         # print(f"h_X_A1: {h_X_A1}")
    #                         # print(f"h_Y_A1: {h_Y_A1}")
    #                         # print(f"h_X_C1: {h_X_C1}")
    #                         # print(f"h_Y_C1: {h_Y_C1}")
    #
    #                         I_A1_L3[jjj] = L3 * 2 * dtheta13 * np.sum(L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
    #                         I_C1_L3[jjj] = L3 * 2 * dtheta13 * np.sum(L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))
    #
    #                         # print(f"I_A1_L3: {I_A1_L3[jjj]}")
    #                         # print(f"I_C1_L3: {I_C1_L3[jjj]}")
    #
    #                     I_A1_L1[iii] += L1 * 2 * dTheta12 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_A1_L3).integral(30, 4000)
    #                     I_C1_L1[iii] += L1 * 2 * dTheta12 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_C1_L3).integral(30, 4000)
    #         A1[lll] = -1 / (A[lll] * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(30, 4000)
    #         C1[lll] = 1 / (A[lll] * 2 * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(30, 4000)
    #     return A1, C1


    def bias(self, XY, Ls, N_L1=20, N_L3=20, Ntheta12=50, Ntheta13=10):
        A = self._qe.normalisation_vector(XY, Ls, dL=2)
        print(f"A: {A}")
        samp1_1 = np.arange(30, 40, 5)
        samp2_1 = np.logspace(1, 3, N_L1) * 4
        Ls1 = np.concatenate((samp1_1, samp2_1))
        samp1_2 = np.arange(30, 40, 5)
        samp2_2 = np.logspace(1, 3, N_L3) * 4
        Ls3 = np.concatenate((samp1_2, samp2_2))
        dTheta1 = np.pi / Ntheta12
        thetas1 = np.arange(dTheta1, np.pi + dTheta1, dTheta1, dtype=float)
        dtheta3 = np.pi / Ntheta13
        thetas3 = np.arange(dtheta3, np.pi + dtheta3, dtheta3, dtype=float)
        N_Ls = np.size(Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        A1 = np.zeros(np.shape(Ls))
        C1 = np.zeros(np.shape(Ls))
        for lll, L in enumerate(Ls):
            print(L)
            I_A1_L1 = np.zeros(np.size(Ls1))
            I_C1_L1 = np.zeros(np.size(Ls1))
            for iii, L1 in enumerate(Ls1):
                for theta1 in thetas1:
                    L_vec = vector.obj(rho=L, phi=0)
                    L1_vec = vector.obj(rho=L1, phi=theta1)
                    L2_vec = L_vec - L1_vec
                    L2 = L2_vec.rho
                    w2 = 0 if (L2 < 3 or L2 > 6000) else 1
                    bi = w2 * 4 * self.mixed_bispectrum("theory", L1, L2, theta1) / (L1 ** 2 * L2 ** 2)
                    I_A1_L3 = np.zeros(np.size(Ls3))
                    I_C1_L3 = np.zeros(np.size(Ls3))
                    if w2 != 0:
                        for jjj, L3 in enumerate(Ls3):
                            L3_vec = vector.obj(rho=L3, phi=thetas3)
                            L5_vec = L1_vec - L3_vec
                            Ls5 = L5_vec.rho
                            w5 = np.ones(np.shape(Ls5))
                            w5[Ls5 < 3] = 0
                            w5[Ls5 > 6000] = 0
                            Xbar_Ybar = XY.replace("B", "E")
                            X_Ybar = XY[0] + XY[1].replace("B","E")
                            Xbar_Y = XY[0].replace("B","E") + XY[1]
                            C_L5 = self._qe.cmb[Xbar_Ybar].Cl_spline(Ls5)
                            C_Ybar_L3 = self._qe.cmb[X_Ybar].Cl_spline(L3)
                            C_Xbar_L3 = self._qe.cmb[Xbar_Y].Cl_spline(L3)
                            L_A1_fac = (L5_vec @ L1_vec) * (L5_vec @ L2_vec)
                            L_C1_fac = (L3_vec @ L1_vec) * (L3_vec @ L2_vec)
                            g_XY = self._qe.weight_function_vector(XY, L3_vec, L_vec)
                            g_YX = self._qe.weight_function_vector(XY[::-1], L3_vec, L_vec)
                            L4_vec = L_vec - L3_vec
                            h_X_A1 = self._qe.geo_fac(XY[0], theta12=L5_vec.deltaphi(L4_vec))
                            h_Y_A1 = self._qe.geo_fac(XY[1], theta12=L5_vec.deltaphi(L3_vec))
                            h_X_C1 = self._qe.geo_fac(XY[0], theta12=L3_vec.deltaphi(L4_vec))
                            h_Y_C1 = self._qe.geo_fac(XY[1], theta12=L3_vec.deltaphi(L4_vec))

                            I_A1_L3[jjj] = L3 * 2 * dtheta3 * np.sum(w5 *L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
                            I_C1_L3[jjj] = L3 * 2 * dtheta3 * np.sum(L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))

                        I_A1_L1[iii] += L1 * 2 * dTheta1 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_A1_L3).integral(30, 4000)
                        I_C1_L1[iii] += L1 * 2 * dTheta1 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_C1_L3).integral(30, 4000)
            A1[lll] = -1 / (A[lll] * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(30, 4000)
            C1[lll] = 1 / (A[lll] * 2 * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(30, 4000)
        return A1, C1


    def bias_phi(self, XY, Ls, N_L1=20, N_L3=20, Ntheta12=50, Ntheta13=10):
        A = self._qe.normalisation(XY, Ls, dL=2)
        print(f"A: {A}")
        samp1_1 = np.arange(30, 40, 5)
        samp2_1 = np.logspace(1, 3, N_L1) * 4
        Ls1 = np.concatenate((samp1_1, samp2_1))
        samp1_2 = np.arange(30, 40, 5)
        samp2_2 = np.logspace(1, 3, N_L3) * 4
        Ls3 = np.concatenate((samp1_2, samp2_2))
        dTheta12 = np.pi / Ntheta12
        thetas12 = np.arange(dTheta12, np.pi + dTheta12, dTheta12, dtype=float)
        dtheta13 = np.pi / Ntheta13
        thetas13 = np.arange(dtheta13, np.pi + dtheta13, dtheta13, dtype=float)
        N_Ls = np.size(Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        A1 = np.zeros(np.shape(Ls))
        C1 = np.zeros(np.shape(Ls))
        for lll, L in enumerate(Ls):
            print(L)
            I_A1_L1 = np.zeros(np.size(Ls1))
            I_C1_L1 = np.zeros(np.size(Ls1))
            for iii, L1 in enumerate(Ls1):
                for theta12 in thetas12:
                    theta_L2 = np.arcsin(L1 * np.sin(theta12) / L)  # Unsure
                    theta_1L = theta12 - theta_L2
                    L2 = self._get_third_L(L1, L, theta_1L)
                    w2 = 0 if (L2 < 3 or np.isnan(L2) or L2 > 6000) else 1
                    bi = w2 * 4 * self.mixed_bispectrum("conv", L1, L2, theta12) / (L1 ** 2 * L2 ** 2)
                    I_A1_L3 = np.zeros(np.size(Ls3))
                    I_C1_L3 = np.zeros(np.size(Ls3))
                    if w2 != 0:
                        for jjj, L3 in enumerate(Ls3):
                            Ls5 = self._get_third_L(L1, L3, thetas13)
                            w5 = np.ones(np.shape(Ls5))
                            w5[Ls5 < 3] = 0
                            w5[Ls5 > 6000] = 0
                            Xbar_Ybar = XY.replace("B", "E")
                            X_Ybar = XY[0] + XY[1].replace("B","E")
                            Xbar_Y = XY[0].replace("B","E") + XY[1]
                            C_L5 = w5 * self._qe.cmb[Xbar_Ybar].Cl_spline(Ls5)
                            C_Ybar_L3 = w5 * self._qe.cmb[X_Ybar].Cl_spline(L3)
                            C_Xbar_L3 = w5 * self._qe.cmb[Xbar_Y].Cl_spline(L3)
                            L1_L1 = L1 ** 2
                            L1_L3 = L1 * L3 * np.cos(thetas13)
                            L1_L2 = L1 * L2 * np.cos(theta12)
                            L2_L3 = L2 * L3 * np.cos(theta12 - thetas13)
                            L_A1_fac = (L1_L1 - L1_L3) * (L1_L2 - L2_L3)
                            L_C1_fac = L1_L3 * L2_L3
                            thetas_L3 = thetas13 - theta_1L
                            g_XY = self._qe.weight_function(XY, L3, L, thetas_L3, curl=False)
                            g_XY[np.isnan(g_XY)] = 0
                            g_YX = self._qe.weight_function(XY[::-1], L3, L, thetas_L3, curl=False)
                            g_YX[np.isnan(g_YX)] = 0
                            # print(f"thetas_L3: {thetas_L3}")
                            # print(f"g_XY: {g_XY}")
                            # print(f"g_YX: {g_YX}")

                            thetas_51 = np.arcsin(L3*np.sin(thetas13)/Ls5)
                            thetas_53 = thetas_51 + thetas13

                            Ls4 = self._get_third_L(L3, L, thetas_L3)
                            # print(f"Ls4: {Ls4}")
                            thetas_4L = np.arcsin(L3*np.sin(thetas13)/Ls4)
                            # print(f"thetas_4L: {thetas_4L}")
                            thetas_34 = 2*np.pi - thetas_4L - thetas_L3
                            # print(f"thetas_34: {thetas_34}")
                            thetas_54 = thetas_34 + thetas_53
                            # print(f"thetas_54: {thetas_54}")

                            h_X_A1 = self._qe.geo_fac(XY[0], theta12=thetas_54)
                            h_X_A1[np.isnan(h_X_A1)] = 0
                            h_Y_A1 = self._qe.geo_fac(XY[1], theta12=thetas_53)
                            h_Y_A1[np.isnan(h_Y_A1)] = 0
                            h_X_C1 = self._qe.geo_fac(XY[0], theta12=thetas_34)
                            h_X_C1[np.isnan(h_X_C1)] = 0
                            h_Y_C1 = self._qe.geo_fac(XY[1], theta12=thetas_34)
                            h_Y_C1[np.isnan(h_Y_C1)] = 0
                            # print(f"h_X_A1: {h_X_A1}")
                            # print(f"h_Y_A1: {h_Y_A1}")
                            # print(f"h_X_C1: {h_X_C1}")
                            # print(f"h_Y_C1: {h_Y_C1}")

                            I_A1_L3[jjj] = L3 * 2 * dtheta13 * np.sum(L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
                            I_C1_L3[jjj] = L3 * 2 * dtheta13 * np.sum(L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))

                            # print(f"I_A1_L3: {I_A1_L3[jjj]}")
                            # print(f"I_C1_L3: {I_C1_L3[jjj]}")

                        I_A1_L1[iii] += L1 * 2 * dTheta12 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_A1_L3).integral(30, 4000)
                        I_C1_L1[iii] += L1 * 2 * dTheta12 * np.sum(bi) * InterpolatedUnivariateSpline(Ls3, I_C1_L3).integral(30, 4000)
            A1[lll] = -1 / (A[lll] * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(30, 4000)
            C1[lll] = 1 / (A[lll] * 2 * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(30, 4000)
        return A1, C1

    def bias_gmv(self, Ls, N_L1=20, N_L3=20, Ntheta12=50, Ntheta13=10):
        A = self._qe.gmv_normalisation(Ls, dL=2)
        samp1_1 = np.arange(30, 40, 5)
        samp2_1 = np.logspace(1, 3, N_L1) * 4
        Ls1 = np.concatenate((samp1_1, samp2_1))
        samp1_2 = np.arange(30, 40, 5)
        samp2_2 = np.logspace(1, 3, N_L3) * 4
        Ls3 = np.concatenate((samp1_2, samp2_2))
        dTheta12 = np.pi / Ntheta12
        thetas12 = np.arange(dTheta12, np.pi + dTheta12, dTheta12, dtype=float)
        dtheta13 = np.pi / Ntheta13
        thetas13 = np.arange(dtheta13, np.pi + dtheta13, dtheta13, dtype=float)
        N_Ls = np.size(Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        A1 = np.zeros(np.shape(Ls))
        C1 = np.zeros(np.shape(Ls))
        for lll, L in enumerate(Ls):
            print(L)
            XYs = ["TT", "TE", "EE", "TB", "EB"]
            for XY in XYs:
                sym_fac = 1 if XY[0] == XY[1] else 2
                I_A1_L1 = np.zeros(np.size(Ls1))
                I_C1_L1 = np.zeros(np.size(Ls1))
                for iii, L1 in enumerate(Ls1):
                    for theta12 in thetas12:
                        theta_L2 = np.arcsin(L1 * np.sin(theta12) / L)  # Unsure
                        theta_1L = theta12 - theta_L2
                        L2 = self._get_third_L(L1, L, theta_1L)
                        w2 = 0 if (L2 < 3 or np.isnan(L2) or L2 > 6000) else 1
                        bi = w2 * 4 * self.mixed_bispectrum("theory", L1, L2, theta12) / (L1 ** 2 * L2 ** 2)
                        I_A1_L3 = np.zeros(np.size(Ls3))
                        I_C1_L3 = np.zeros(np.size(Ls3))
                        if w2 != 0:
                            for jjj, L3 in enumerate(Ls3):
                                Ls5 = self._get_third_L(L1, L3, thetas13)
                                w5 = np.ones(np.shape(Ls5))
                                w5[Ls5 < 3] = 0
                                w5[Ls5 > 6000] = 0
                                Xbar_Ybar = XY.replace("B", "E")
                                X_Ybar = XY[0] + XY[1].replace("B","E")
                                Xbar_Y = XY[0].replace("B","E") + XY[1]
                                C_L5 = w5 * self._qe.cmb[Xbar_Ybar].Cl_spline(Ls5)
                                C_Ybar_L3 = w5 * self._qe.cmb[X_Ybar].Cl_spline(L3)
                                C_Xbar_L3 = w5 * self._qe.cmb[Xbar_Y].Cl_spline(L3)
                                L1_L1 = L1 ** 2
                                L1_L3 = L1 * L3 * np.cos(thetas13)
                                L1_L2 = L1 * L2 * np.cos(theta12)
                                L2_L3 = L2 * L3 * np.cos(theta12 - thetas13)
                                L_A1_fac = (L1_L1 - L1_L3) * (L1_L2 - L2_L3)
                                L_C1_fac = L1_L3 * L2_L3
                                thetas_L3 = thetas13 - theta_1L
                                g_XY = self._qe.gmv_weight_function(XY, L3, L, thetas_L3, curl=True)
                                g_YX = self._qe.gmv_weight_function(XY[::-1], L3, L, thetas_L3, curl=True)

                                thetas_51 = np.arcsin(L3*np.sin(thetas13)/Ls5)
                                thetas_53 = thetas_51 + thetas13

                                Ls4 = self._get_third_L(L3, L, thetas_L3)
                                thetas_4L = np.arcsin(L3*np.sin(thetas13)/Ls4)
                                thetas_34 = 2*np.pi - thetas_4L - thetas_L3
                                thetas_54 = thetas_34 + thetas_53

                                h_X_A1 = self._qe.geo_fac(XY[0], theta12=thetas_54)
                                h_Y_A1 = self._qe.geo_fac(XY[1], theta12=thetas_53)
                                h_X_C1 = self._qe.geo_fac(XY[0], theta12=thetas_34)
                                h_Y_C1 = self._qe.geo_fac(XY[1], theta12=thetas_34)

                                I_A1_L3[jjj] = L3 * np.sum(2 * dtheta13 * L_A1_fac * C_L5 * g_XY * h_X_A1 * h_Y_A1)
                                I_C1_L3[jjj] = L3 * np.sum(2 * dtheta13 * L_C1_fac * ((C_Ybar_L3 * g_XY * h_Y_C1) + (C_Xbar_L3 * g_YX * h_X_C1)))
                            I_A1_L1[iii] += sym_fac * L1 * np.sum(2 * dTheta12 * bi * InterpolatedUnivariateSpline(Ls3, I_A1_L3).integral(30, 4000))
                            I_C1_L1[iii] += sym_fac * L1 * np.sum(2 * dTheta12 * bi * InterpolatedUnivariateSpline(Ls3, I_C1_L3).integral(30, 4000))
                A1[lll] += -1 / (A[lll] * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_A1_L1).integral(30, 4000)
                C1[lll] += 1 / (A[lll] * 2 * (2 * np.pi) ** 4) * InterpolatedUnivariateSpline(Ls1, I_C1_L1).integral(30, 4000)
        return A1, C1

