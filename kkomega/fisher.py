import numpy as np
from bispectra import Bispectra
from powerspectra import Powerspectra
from noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
import copy

class Fisher:
    """

    """

    def __init__(self, N0_file, ell_file, M_file, N0_offset=0, N0_ell_factors=True):
        """

        Parameters
        ----------
        N0_file
        ell_file
        M_file
        """
        self.noise = Noise(N0_file, N0_offset)
        self.N0_ell_factors = N0_ell_factors
        ells_sample = np.load(ell_file)
        M_matrix = np.load(M_file)
        self.bi = Bispectra(M_spline=True, ells_sample=ells_sample, M_matrix=M_matrix)

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def _get_Cl_kappa(self,ellmax):
        power = Powerspectra()
        ells = np.arange(ellmax + 1)
        return power.get_kappa_ps(ells)

    def _get_L3(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample, arr)

    def _get_Ns(self, Lmax, typ, include_N0_kappa, all_splines=False):
        N0_kappa = self.noise.get_N0("phi", Lmax, tidy=True, ell_factors=self.N0_ell_factors)
        Cl_kappa = self._get_Cl_kappa(Lmax)
        if include_N0_kappa == "both":
            C1 = Cl_kappa + N0_kappa
            C2 = Cl_kappa + N0_kappa
        elif include_N0_kappa == "one":
            C1 = Cl_kappa + N0_kappa
            C2 = Cl_kappa
        else:
            C1 = Cl_kappa
            C2 = Cl_kappa
        if typ == "conv_rot":
            N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
            if all_splines:
                C1_spline = self._interpolate(C1)
                C2_spline = self._interpolate(C2)
                return C1_spline, C2_spline, N0_omega_spline
            return C1, C2, N0_omega_spline
        C3 = Cl_kappa + N0_kappa
        C3_spline = self._interpolate(C3)
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C2, C1, C3_spline

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ, include_N0_kappa):
        C1, C2, C3_spline = self._get_Ns(Lmax, typ, include_N0_kappa)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        L3 = self._get_L3(Ls[:, None], Ls[None, :], thetas[:, None, None])
        w = np.ones(np.shape(L3))
        w[L3 < 2] = 0
        w[L3 > Lmax] = 0
        return Ls, L3, dTheta, w, C1, C2, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky, include_N0_kappa):
        Ls, L3, dTheta, w, C1, C2, N3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv_rot", include_N0_kappa)
        bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_rot_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * N3_spline(L3)))
        return I * f_sky / ((2 * np.pi) ** 3)

    def _get_convergence_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky, include_N0_kappa):
        Ls, L3, dTheta, w, C1, C2, C3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv", include_N0_kappa)
        bi_conv = self.bi.get_convergence_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)))
        return I * f_sky / (3 * (2*np.pi) ** 3)

    def _integral_prep_sample(self, Ls, Ntheta, typ, include_N0_kappa):
        Lmax = int(Ls[-1])
        C1_spline, C2_spline, C3_spline = self._get_Ns(Lmax, typ, include_N0_kappa, all_splines=True)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[0] = Ls[0]
        dLs[1:] = Ls[1:] - Ls[0:-1]
        return Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr, include_N0_kappa):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "conv_rot", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_rot_conv ** 2) / (C1_spline(L1) * C2_spline(L2) * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_convergence_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr, include_N0_kappa):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta, "conv", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_conv ** 2) / (C1_spline(L1) * C2_spline(L2) * C3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/(3 * (2*np.pi) ** 3)
        if arr:
            return I
        return np.sum(I)

    def _integral_prep_arr(self, Lmax, dL, Ntheta, typ, include_N0_kappa):
        C1, C2, C3_spline = self._get_Ns(Lmax, typ, include_N0_kappa)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        weights = np.ones(np.shape(thetas))
        return Ls, thetas, dTheta, weights, C1, C2, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv_rot", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv_rot = self.bi.get_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_conv_rot ** 2) / (C1[L1] * C2[L2] * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_convergence_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa):
        Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_conv ** 2) / (C1[L1] * C2[L2] * C3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= f_sky / (12 * np.pi ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def get_convergence_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False, include_N0_kappa="both"):
        """

        Parameters
        ----------
        Lmax
        dL
        Ls
        Ntheta
        f_sky
        arr

        Returns
        -------

        """
        if Ls is None:
            if arr:
                return self._get_convergence_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa)
            else:
                return self._get_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky, include_N0_kappa)
        return self._get_convergence_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr, include_N0_kappa)


    def get_convergence_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False, include_N0_kappa="both"):
        """

        Parameters
        ----------
        Lmax
        dL
        Ls
        Ntheta
        f_sky
        arr

        Returns
        -------

        """
        if Ls is None:
            if arr:
                return self._get_convergence_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa)
            else:
                return self._get_convergence_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky, include_N0_kappa)
        return self._get_convergence_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr, include_N0_kappa)