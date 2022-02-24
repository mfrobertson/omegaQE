import numpy as np


class Noise:

    def __init__(self, file, offset=0):
        self.N0 = np.load(file)
        self.offset = offset

    def _get_N0_phi(self, ellmax):
        return np.concatenate((np.zeros(self.offset), self.N0[0][:ellmax + 1 - self.offset]))

    def _get_N0_curl(self, ellmax):
        return np.concatenate((np.zeros(self.offset), self.N0[1][:ellmax + 1 - self.offset]))

    def _get_N0_kappa(self, ellmax):
        ells = np.arange(ellmax + 1)
        return self._get_N0_phi(ellmax) * 0.25 * (ells + 0.5) ** 4

    def _get_N0_omega(self, ellmax):
        ells = np.arange(ellmax + 1)
        return self._get_N0_curl(ellmax) * 0.25 * (ells + 0.5) ** 4

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > self.offset + 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def get_N0(self, typ="phi", ellmax=4000, tidy=False, ell_factors=False):
        """

        Parameters
        ----------
        typ
        ellmax
        tidy

        Returns
        -------

        """
        if typ == "phi":
            if ell_factors:
                N0 = self._get_N0_kappa(ellmax)
            else:
                N0 = self._get_N0_phi(ellmax)
        elif typ == "curl":
            if ell_factors:
                N0 = self._get_N0_omega(ellmax)
            else:
                N0 = self._get_N0_curl(ellmax)
        if tidy:
            return self._replace_bad_Ls(N0)
        return N0