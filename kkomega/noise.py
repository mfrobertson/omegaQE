import numpy as np


class Noise:

    def __init__(self, file):
        self.N0 = np.load(file)

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

    def get_N0(self, typ="phi", ellmax=4000, tidy=False):
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
            N0 = self._get_N0_phi(ellmax)
        elif typ == "curl":
            N0 = self._get_N0_curl(ellmax)
        elif typ == "kappa":
            N0 = self._get_N0_kappa(ellmax)
        elif typ == "omega":
            N0 = self._get_N0_omega(ellmax)
        if tidy:
            return self._replace_bad_Ls(N0)
        return N0