from fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmology import Cosmology


class Bias:

    class Holder:

        def __init__(self):
            self.F_L_spline = None
            self.typs = None
            self.nu = None
            self.C_inv = None
            self.Cl_kk = None
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


    def __init__(self, N0_file, N0_offset=0, N0_ell_factors=True, M_path=None):
        self._fisher = Fisher(N0_file, N0_offset, N0_ell_factors)
        self.N0_offset = N0_offset
        self.N0_ell_factors = N0_ell_factors
        self._cache = self.Holder()
        self._cosmo = Cosmology()
        if M_path is not None:
            self._fisher.setup_bispectra(M_path, 4000, 100)


    def _build_F_L(self, typs, nu):
        print("F_L_build")
        Ls1 = np.arange(30, 40, 2)
        Ls2 = np.logspace(1, 3, 100) * 4
        sample_Ls = np.concatenate((Ls1, Ls2))
        _, F_L, C_inv = self._fisher.get_F_L(typs, Ls=sample_Ls, Ntheta=100, nu=nu, return_C_inv=True)
        self._cache.Cl_kk = self._fisher.get_Cl("kk", ellmax=4000, nu=nu)
        self._cache.Cl_gk = self._fisher.get_Cl("gk", ellmax=4000, nu=nu)
        self._cache.Cl_Ik = self._fisher.get_Cl("Ik", ellmax=4000, nu=nu)
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

    def _mixed_bi_element(self, typ, typs, L, L1, L2, nu, C_inv):
        bi_typ = typ[:2] + "w"
        p = typ[2]
        q = typ[3]
        bi = self._fisher.bi.get_bispectrum(bi_typ, L1, L2, L, M_spline=True, nu=nu)
        cov_inv1, cov_inv2 = self._get_cov_invs(typ, typs, C_inv)
        Ls = np.arange(np.size(cov_inv1))
        cov_inv1_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv1[1:])
        cov_inv2_spline = InterpolatedUnivariateSpline(Ls[1:], cov_inv2[1:])
        Cl_pk = self._cache.get_Cl(p+"k")
        Cl_qk = self._cache.get_Cl(q+"k")
        Cl_pk_spline = InterpolatedUnivariateSpline(Ls, Cl_pk)
        Cl_qk_spline = InterpolatedUnivariateSpline(Ls, Cl_qk)
        return bi * cov_inv1_spline(L1) * cov_inv2_spline(L2) * (Cl_pk_spline(L1)*Cl_qk_spline(L2) + Cl_pk_spline(L2)*Cl_qk_spline(L1))


    def _mixed_bispectrum(self, typs, L, L1, L2, nu):
        F_L = self._cache.F_L_spline(L)
        typs = np.char.array(typs)
        C_inv = self._cache.C_inv
        # C_inv = self._fisher.get_C_inv(typs, Lmax=4000, nu=nu)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        perms = 0
        mixed_bi = None
        for iii in np.arange(Ncombos):
            for jjj in np.arange(Ncombos):
                typ = combos[iii] + combos[jjj]
                mixed_bi_element = self._mixed_bi_element(typ, typs, L, L1, L2, nu, C_inv)
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
        return 2/(F_L * L1**2 * L2**2) * mixed_bi

    def mixed_bispectrum(self, typs, L, L1, L2, nu=353e9):
        """

        Parameters
        ----------
        typs
        L
        L1
        L2
        nu

        Returns
        -------

        """
        if typs == "theory":
            return 4*self._fisher.bi.get_bispectrum("kkw", L1, L2, L)/(L1**2 * L2**2)
        if self._cache is None or self._cache.typs != typs or self._cache.nu != nu:
            self._build_F_L(typs, nu)
        return self._mixed_bispectrum(list(typs), L, L1, L2, nu)

    def _response(self, L1, L2, theta):
        L3 = self._get_third_L(L1,L2,theta)
        w = np.ones(np.shape(L3))
        w[L3 < 30] = 0
        return w*L1*L2*np.sin(theta)*(self._cache.Cl_gradT_spline(L1) + self._cache.Cl_gradT_spline(L3))

    def _weight_function(self, L1, L2, theta):
        L3 = self._get_third_L(L1,L2,theta)
        w = np.ones(np.shape(L3))
        w[L3<30]=0
        denom = 2*(self._cache.Cl_TT_spline(L1)+self._cache.N_T_spline(L1))*(self._cache.Cl_TT_spline(L3)+self._cache.N_T_spline(L3))
        return w*self._response(L1, L2, theta)/denom

    def normalisation(self, Ls, dL=10, Ntheta=100):
        """

        Parameters
        ----------
        Ls
        dL
        Ntheta

        Returns
        -------

        """
        Lmax = 4000
        Lmin = 30
        L_prims = np.arange(Lmin, Lmax+1, dL)[None, :]
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)[:,None]
        if self._cache.Cl_TT_spline is None:
            Cl_T_lens = self._cosmo.get_TT_lens_ps(Lmax)
            Cl_T_lens_spline = InterpolatedUnivariateSpline(np.arange(np.size(Cl_T_lens)), Cl_T_lens)
            Cl_gradT_lens = self._cosmo.get_TT_grad_lens_ps(Lmax)
            Cl_gradT_lens_spline = InterpolatedUnivariateSpline(np.arange(np.size(Cl_gradT_lens)), Cl_gradT_lens)
            N_T = self._fisher.noise.get_cmb_gaussian_N("T", ellmax=Lmax)
            N_T_spline = InterpolatedUnivariateSpline(np.arange(np.size(N_T)), N_T)
            self._cache.Cl_TT_spline = Cl_T_lens_spline
            self._cache.Cl_gradT_spline = Cl_gradT_lens_spline
            self._cache.N_T_spline = N_T_spline
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        for iii, L in enumerate(Ls):
            g = self._weight_function(L_prims, L, thetas)
            f = self._response(L_prims, L, thetas)
            I = 2 * dTheta * dL *np.sum(L_prims * g * f)
            A[iii] = I/((2*np.pi)**2)
        return A

    def _get_third_L(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def N_A1(self, typs, L, dL=10, Ntheta=100, nu=353e9):
        """

        Parameters
        ----------
        typs
        L
        dL
        Ntheta
        nu

        Returns
        -------

        """
        if self._cache is None:
            self._build_F_L(typs, nu)
        A = self.normalisation(L)
        Ls1 = np.arange(30, 4000, dL)
        L_prims = np.arange(30, 4000, dL)
        dTheta = np.pi / Ntheta
        thetas1 = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        Ntheta_prim = 10
        dtheta_prim = np.pi / Ntheta_prim
        thetas_prim = np.arange(dtheta_prim, np.pi + dtheta_prim, dtheta_prim, dtype=float)
        I_L1 = 0
        for L1 in Ls1:
            print(L1)
            I_theta1 = 0
            for theta1 in thetas1:
                L2 = self._get_third_L(L, L1, theta1)
                w=0 if L2 < 30 else 1
                bi = 4*self._fisher.bi.get_bispectrum("kkw", L1, L2, L)/(L1**2 * L2**2)
                I_L_prim = 0
                for L_prim in L_prims:
                    g = self._weight_function(L_prim, L, thetas_prim)
                    thetas_1_prim = theta1 - thetas_prim
                    L_fac = (L1**2 - (L_prim*L1*np.cos(thetas_1_prim))) * ((L1*L*np.cos(theta1)) - L1**2 - (L_prim*L*np.cos(thetas_prim)) + (L_prim*L1*np.cos(thetas_1_prim)))
                    L1_minus_L_prim = self._get_third_L(L1, L_prim, thetas_1_prim)
                    w1 = np.ones(np.shape(L1_minus_L_prim))
                    w1[L1_minus_L_prim < 30] = 0
                    C_TT = self._cache.Cl_TT_spline(L1_minus_L_prim)
                    I_theta_prim = 2 * dtheta_prim * np.sum(g*L_fac*C_TT*w1)
                    I_L_prim += L_prim * dL * I_theta_prim
                I_theta1 += 2 * dTheta * I_L_prim * bi * w
            I_L1 += L1 * dL * I_theta1
        return -1/A * I_L1/((2*np.pi)**2)

    # def N_A1_vec(self, typs, Ls, dL_prim=10, Ntheta=100, Ntheta_prim=50, nu=353e9):
    #     """
    #
    #     Parameters
    #     ----------
    #     typs
    #     Ls
    #     dL_prim
    #     Ntheta
    #     Ntheta_prim
    #     nu
    #
    #     Returns
    #     -------
    #
    #     """
    #     if self._cache is None:
    #         self._build_F_L(typs, nu)
    #     A = self.normalisation(Ls)
    #     Lmin=30
    #     Lmax=4000
    #     samp1 = np.arange(30, 40, 2)
    #     samp2 = np.logspace(1, 3, 100) * 4
    #     Ls1 = np.concatenate((samp1, samp2))
    #     L_prims = np.arange(Lmin, Lmax+1, dL_prim)
    #     dTheta = np.pi / Ntheta
    #     thetas1 = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
    #     dtheta_prim = np.pi / Ntheta_prim
    #     thetas_prim = np.arange(dtheta_prim, np.pi + dtheta_prim, dtheta_prim, dtype=float)
    #     N_Ls = np.size(Ls)
    #     if N_Ls == 1:
    #         Ls = np.ones(1)*Ls
    #     N_A1 = np.zeros(np.size(Ls))
    #     for iii, L in enumerate(Ls):
    #         I = np.zeros(np.size(Ls1))
    #         for jjj, L1 in enumerate(Ls1):
    #             L2 = self._get_third_L(L, L1, thetas1)
    #             w = np.ones(np.shape(L2))
    #             w[L2 < Lmin] = 0
    #             bi = self.mixed_bispectrum(typs, L, L1, L2, nu)
    #             bi[np.isnan(bi)]=0
    #             g = self._weight_function(L_prims[None, :, None], L, thetas_prim[:, None, None])
    #             thetas_1_prim = thetas1[None,None,:] - thetas_prim[:, None, None]
    #             L_fac = (L1 ** 2 - (L_prims[None, :, None] * L1 * np.cos(thetas_1_prim))) * (
    #                         (L1 * L * np.cos(thetas1[None,None,:])) - L1 ** 2 - (
    #                             L_prims[None, :, None] * L * np.cos(thetas_prim[:, None, None])) + (
    #                                     L_prims[None, :, None] * L1 * np.cos(thetas_1_prim)))
    #             L1_minus_L_prim = self._get_third_L(L1, L_prims[None, :, None], thetas_1_prim)
    #             w1 = np.ones(np.shape(L1_minus_L_prim))
    #             w1[L1_minus_L_prim < Lmin] = 0
    #             C_TT = self._cache.Cl_TT_spline(L1_minus_L_prim)
    #
    #             I[jjj] = L1 * 2 * dTheta * 2 * dL_prim * dtheta_prim * np.sum(L_prims[None, :, None] * g * L_fac * C_TT * w1 * bi[None,None,:] * w[None,None,:])
    #         N_A1[iii] = -1 / A[iii] * InterpolatedUnivariateSpline(Ls1, I).integral(Lmin, Lmax) / ((2 * np.pi) ** 2)
    #     return N_A1

    def bias_vec(self, typs, Ls, dL_prim=10, Ntheta=100, Ntheta_prim=50, nu=353e9):
        """

        Parameters
        ----------
        typs
        Ls
        dL_prim
        Ntheta
        Ntheta_prim
        nu

        Returns
        -------

        """
        if self._cache is None:
            self._build_F_L(typs, nu)
        A = self.normalisation(Ls)
        Lmin=30
        Lmax=4000
        samp1 = np.arange(30, 60, 3)
        samp2 = np.logspace(1, 3, 100) * 6
        Ls1 = np.concatenate((samp1, samp2))
        L_prims = np.arange(Lmin, 6000+1, dL_prim)
        dTheta = np.pi / Ntheta
        thetas1 = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        dtheta_prim = np.pi / Ntheta_prim
        thetas_prim = np.arange(dtheta_prim, np.pi + dtheta_prim, dtheta_prim, dtype=float)
        N_Ls = np.size(Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        N_A1 = np.zeros(np.size(Ls))
        N_C1 = np.zeros(np.size(Ls))
        for iii, L in enumerate(Ls):
            I_A1 = np.zeros(np.size(Ls1))
            I_C1 = np.zeros(np.size(Ls1))
            for jjj, L1 in enumerate(Ls1):
                L2 = self._get_third_L(L, L1, thetas1)
                w = np.ones(np.shape(L2))
                w[L2 < Lmin] = 0
                bi = self.mixed_bispectrum(typs, L, L1, L2, nu)
                bi[np.isnan(bi)]=0
                g = self._weight_function(L_prims[None, :, None], L, thetas_prim[:, None, None])
                thetas_1_prim = thetas1[None,None,:] - thetas_prim[:, None, None]
                L_fac_A1 = (L1 ** 2 - (L_prims[None, :, None] * L1 * np.cos(thetas_1_prim))) * (
                            (L1 * L * np.cos(thetas1[None,None,:])) - L1 ** 2 - (
                                L_prims[None, :, None] * L * np.cos(thetas_prim[:, None, None])) + (
                                        L_prims[None, :, None] * L1 * np.cos(thetas_1_prim)))
                L_prims_dot_L1 = L_prims[None, :, None]*L1*np.cos(thetas_1_prim)
                L_fac_C1 = (L_prims[None, :, None]*L*np.cos(thetas_prim[:, None, None])*L_prims_dot_L1) - L_prims_dot_L1**2
                L1_minus_L_prim = self._get_third_L(L1, L_prims[None, :, None], thetas_1_prim)
                w1 = np.ones(np.shape(L1_minus_L_prim))
                w1[L1_minus_L_prim < Lmin] = 0
                C_TT_A1 = self._cache.Cl_TT_spline(L1_minus_L_prim)
                C_TT_C1 = self._cache.Cl_TT_spline(L_prims[None, :, None])

                I_A1[jjj] = L1 * 2 * dTheta * 2 * dL_prim * dtheta_prim * np.sum(L_prims[None, :, None] * g * L_fac_A1 * C_TT_A1 * w1 * bi[None,None,:] * w[None,None,:])
                I_C1[jjj] = L1 * 2 * dTheta * 2 * dL_prim * dtheta_prim * np.sum(L_prims[None, :, None] * g * L_fac_C1 * 2*C_TT_C1 * w1 * bi[None,None,:] * w[None,None,:])
            N_A1[iii] = -1 / A[iii] * InterpolatedUnivariateSpline(Ls1, I_A1).integral(Lmin, Lmax) / ((2 * np.pi) ** 2)
            N_C1[iii] = -1 / (2*A[iii]) * InterpolatedUnivariateSpline(Ls1, I_C1).integral(Lmin, Lmax) / ((2 * np.pi) ** 2)

        return N_A1, N_C1



