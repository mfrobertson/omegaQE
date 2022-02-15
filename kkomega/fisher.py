import numpy as np
from bispectra import Bispectra
from powerspectra import Powerspectra
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
        self.N0 = np.load(N0_file)
        ells_sample = np.load(ell_file)
        M_matrix = np.load(M_file)
        self.bi = Bispectra(M_spline=True, ells_sample=ells_sample, M_matrix=M_matrix)

    def _get_N0_phi(self, ellmax):
        return self.N0[0][:ellmax + 1]

    def _get_N0_curl(self, ellmax):
        return self.N0[1][:ellmax + 1]

    def _get_N0_kappa(self, ellmax):
        ells = np.arange(0, ellmax + 1)
        return self._get_N0_phi(ellmax) * 0.25 * (ells + 0.5) ** 4

    def _get_N0_omega(self, ellmax):
        ells = np.arange(0, ellmax + 1)
        return self._get_N0_curl(ellmax) * 0.25 * (ells + 0.5) ** 4

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
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta)))

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample, arr)

    def get_convergence_rotation_bispectrum_Fisher(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1):
        N0_omega_spline = self._interpolate(self._get_N0_omega(L3max))
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(Lmax))
        Cl_kappa = self._get_Cl_kappa(Lmax)
        C = Cl_kappa + N0_kappa
        I = 0
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        weights = np.ones(np.size(thetas))
        Ls = np.arange(2, Lmax + 1, dL)
        for L1 in Ls:
            for L2 in Ls:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3>L3max] = 0
                w[L3 < 2] = 0
                bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I += 2 * 2 * np.pi * dL * dL * L1 * L2 * dTheta * np.dot(w, (bi_rot_conv ** 2) / (C[L1] * C[L2] * N0_omega_spline(L3)))
        return I*f_sky/((2*np.pi)**3)

    def get_convergence_bispectrum_Fisher(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1):
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(L3max))
        Cl_kappa = self._get_Cl_kappa(L3max)
        C = Cl_kappa + N0_kappa
        C_spline = self._interpolate(C)
        I = 0
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        weights = np.ones(np.size(thetas))
        Ls = np.arange(2, Lmax + 1, dL)
        for L1 in Ls:
            for L2 in Ls:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > L3max] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I += 2 * 2 * np.pi * dL * dL * L1 * L2 * np.dot(w, dTheta * (bi_conv ** 2) / (C[L1] * C[L2] * C_spline(L3)))
        return I * f_sky / (3 * (2 * np.pi) ** 3)

    def get_convergence_rotation_bispectrum_Fisher2(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1, arr=False):
        """

        Parameters
        ----------
        Lmax
        dL
        dTheta
        L3max
        f_sky

        Returns
        -------

        """
        N0_omega_spline = self._interpolate(self._get_N0_omega(L3max))
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(Lmax))
        Cl_kappa = self._get_Cl_kappa(Lmax)
        C = Cl_kappa + N0_kappa
        thetas = np.arange(dTheta, np.pi, dTheta, dtype=float)
        weights = np.ones(np.size(thetas))
        Ls = np.arange(2, Lmax + 1, dL)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > L3max] = 0
                w[L3 < 2] = 0
                bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += 2 * 2 * 2 * np.pi *dL * dL * L1 * L2 * dTheta * np.dot(w, (bi_rot_conv ** 2) / (C[L1] * C[L2] * N0_omega_spline(L3)))
            I[iii] = I_tmp
        I *= f_sky/((2*np.pi)**3)
        if arr:
            return Ls, I
        return np.sum(I)

    def get_convergence_bispectrum_Fisher2(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1, arr=False):
        """

        Parameters
        ----------
        Lmax
        dL
        dTheta
        L3max
        f_sky

        Returns
        -------

        """
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(L3max))
        Cl_kappa = self._get_Cl_kappa(L3max)
        C = Cl_kappa + N0_kappa
        C_spline = self._interpolate(C)
        thetas = np.arange(dTheta, np.pi, dTheta, dtype=float)
        weights = np.ones(np.size(thetas))
        Ls = np.arange(2, Lmax + 1, dL)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > L3max] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += 2 * 2 * np.pi * dL * dL * L1 * L2 * np.dot(w, dTheta * (bi_conv ** 2) / (C[L1] * C[L2] * C_spline(L3)))
            I[iii] = I_tmp
        I *= f_sky / (12 * np.pi ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def get_convergence_rotation_bispectrum_Fisher3(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1):
        N0_omega_spline = self._interpolate(self._get_N0_omega(L3max))
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(Lmax))
        Cl_kappa = self._get_Cl_kappa(Lmax)
        C = Cl_kappa + N0_kappa
        thetas = np.arange(dTheta, np.pi, dTheta, dtype=float)
        Ls = np.arange(2, Lmax + 1, dL)
        L3 = self._get_L3(Ls[:,None], Ls[None,:], thetas[:,None,None])
        w = np.ones(np.shape(L3))
        w[L3 < 2] = 0
        w[L3 > L3max] = 0
        bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_rot_conv ** 2)/ (C[None, Ls, None] * C[None, None, Ls] * N0_omega_spline(L3)))
        return I * f_sky / ((2 * np.pi) ** 3)

    def get_convergence_bispectrum_Fisher3(self, Lmax, dL=1, dTheta=0.3, L3max=4000, f_sky=1):
        N0_kappa = self._replace_bad_Ls(self._get_N0_kappa(L3max))
        Cl_kappa = self._get_Cl_kappa(L3max)
        C = Cl_kappa + N0_kappa
        C_spline = self._interpolate(C)
        thetas = np.arange(dTheta, np.pi, dTheta, dtype=float)
        Ls = np.arange(2, Lmax + 1, dL)
        L3 = self._get_L3(Ls[:,None], Ls[None,:], thetas[:,None,None])
        w = np.ones(np.shape(L3))
        w[L3 < 2] = 0
        w[L3 > L3max] = 0
        bi_conv = self.bi.get_convergence_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_conv ** 2)/ (C[None, Ls, None] * C[None, None, Ls] * C_spline(L3)))
        return I * f_sky / (3 * (2*np.pi) ** 3)