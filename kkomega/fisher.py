import numpy as np
from bispectra import Bispectra

class Fisher:

    def __init__(self, N0_file, ell_file, M_file):
        self.N0 = np.load(N0_file)
        ells_sample = np.load(ell_file)
        M_matrix = np.load(M_file)
        self.bi = Bispectra(M_spline=True, ells_sample=ells_sample, M_matrix=M_matrix)

    def _triangle_third_length(self, L1, L2, steps, Lmax):
        Lmin = np.floor(np.abs(L1 - L2)) + 1
        Lmax_triangle = np.ceil(L1 + L2)
        if Lmax > Lmax_triangle:
            Lmax = Lmax_triangle
        return np.arange(Lmin, Lmax, steps, dtype=int)

    def _L3_condensing_for_Fisher(self, L3s, Lmax):
        return L3s[np.where(L3s <= Lmax)]

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

    def _interpolate_N0(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > 1:
                print(f"N0[{L}] = nan, replacing with mean of N[{L-1}] = {N0[L-1]} and N[{L+1}] = {N0[L+1]}")
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        print("-------------------------------------")
        return N0

    def convergence_rotation_bispectrum_SN(self, Lmax, dL=1, L3max=4000, f_sky=1):
        N0_omega = self._interpolate_N0(self._get_N0_omega(L3max))
        N0_kappa = self._interpolate_N0(self._get_N0_kappa(Lmax))
        Ls = np.arange(2, Lmax + 1, dL)
        I = 0
        for L1 in Ls:
            for L2 in Ls:
                L3 = self._triangle_third_length(L1, L2, dL, L3max)
                bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                N_L1 = N0_kappa[L1]
                N_L2 = N0_kappa[L2]
                N_L3 = N0_omega[L3]
                I += dL * np.sum((bi_rot_conv**2)/(N_L1*N_L2*N_L3))
        return I*f_sky/((2*np.pi)**3)

    def convergence_bispectrum_SN(self, Lmax, dL=1, L3max=4000, f_sky=1):
        N0_kappa = self._interpolate_N0(self._get_N0_kappa(L3max))
        Ls = np.arange(2, Lmax + 1, dL)
        I = 0
        for L1 in Ls:
            for L2 in Ls:
                L3 = self._triangle_third_length(L1, L2, dL, L3max)
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                N_L1 = N0_kappa[L1]
                N_L2 = N0_kappa[L2]
                N_L3 = N0_kappa[L3]
                I += dL * np.sum((bi_conv ** 2) / (N_L1 * N_L2 * N_L3))
        return I * f_sky / (3 * (2 * np.pi) ** 3)