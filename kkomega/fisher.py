import numpy as np
from bispectra import Bispectra
from powerspectra import Powerspectra
from noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
import copy

class Fisher:
    """

    """

    def __init__(self, N0_file, ell_file, M_file):
        """

        Parameters
        ----------
        N0_file
        ell_file
        M_file
        """
        self.noise = Noise(N0_file)
        ells_sample = np.load(ell_file)
        M_matrix = np.load(M_file)
        self.bi = Bispectra(M_spline=True, ells_sample=ells_sample, M_matrix=M_matrix)

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def _get_Cl_kappa(self, ellmax):
        power = Powerspectra()
        ells = np.arange(0, ellmax + 1)
        return power.get_kappa_ps(ells)

    def _get_L3(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample, arr)

    def _get_Ns(self, Lmax, typ, all_splines=False):
        N0_kappa = self.noise.get_N0("kappa", Lmax, tidy=True)
        Cl_kappa = self._get_Cl_kappa(Lmax)
        C = Cl_kappa + N0_kappa
        if typ == "conv_rot":
            N0_omega_spline = self._interpolate(self.noise.get_N0("omega", Lmax))
            if all_splines:
                C_spline = self._interpolate(C)
                return C_spline, C_spline, N0_omega_spline
            return C, C, N0_omega_spline
        C_spline = self._interpolate(C)
        if all_splines:
            return C_spline, C_spline, C_spline
        return C, C, C_spline

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ):
        C1, C2, C3_spline = self._get_Ns(Lmax, typ)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        L3 = self._get_L3(Ls[:, None], Ls[None, :], thetas[:, None, None])
        w = np.ones(np.shape(L3))
        w[L3 < 2] = 0
        w[L3 > Lmax] = 0
        return Ls, L3, dTheta, w, C1, C2, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_vec(self, Lmax, dL=1, Ntheta=10, f_sky=1):
        Ls, L3, dTheta, w, C1, C2, N3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv_rot")
        bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_rot_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * N3_spline(L3)))
        return I * f_sky / ((2 * np.pi) ** 3)

    def _get_convergence_bispectrum_Fisher_vec(self, Lmax, dL=1, Ntheta=10, f_sky=1):
        Ls, L3, dTheta, w, C1, C2, C3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv")
        bi_conv = self.bi.get_convergence_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)))
        return I * f_sky / (3 * (2*np.pi) ** 3)

    def _integral_prep_sample(self, Ls, Ntheta, typ):
        Lmax = int(Ls[-1])
        C1_spline, C2_spline, C3_spline = self._get_Ns(Lmax, typ, all_splines=True)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[0] = Ls[0]
        dLs[1:] = Ls[1:] - Ls[0:-1]
        return Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta=10, f_sky=1, arr=False):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "conv_rot")
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

    def _get_convergence_bispectrum_Fisher_sample(self, Ls, Ntheta=10, f_sky=1, arr=False):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta, "conv")
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

    def _integral_prep_arr(self, Lmax, dL, Ntheta, typ):
        C1, C2, C3_spline = self._get_Ns(Lmax, typ)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        weights = np.ones(np.shape(thetas))
        return Ls, thetas, dTheta, weights, C1, C2, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_arr(self, Lmax, dL=1, Ntheta=10, f_sky=1, arr=False):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv_rot")
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

    def _get_convergence_bispectrum_Fisher_arr(self, Lmax, dL=1, Ntheta=10, f_sky=1, arr=False):
        Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv")
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

    def get_convergence_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
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
                return self._get_convergence_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                return self._get_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
        return self._get_convergence_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)


    def get_convergence_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
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
                return self._get_convergence_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                return self._get_convergence_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
        return self._get_convergence_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)