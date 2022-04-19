import numpy as np
from cosmology import Cosmology


class Noise:
    """
    Handles CMB experimental noise.

    Attributes
    ----------
    N0 : ndarray
        2D array containing the convergence noise in row 0, and the curl noise in row 1. (The same format as Lensit)
    offset : int
        Essentially the value of ellmin of the N0 array.
    """

    def __init__(self):
        """
        Constructor

        Parameters
        ----------
        file : str
            Path to .npy file containing the convergence noise in row 0, and the curl noise in row 1. (The same format as Lensit)
        offset : int
            Essentially the value of ellmin in the file. If the first column represents ell = 2, set offset to 2.
        """
        self.N0 = None
        self.cmb_offset = None
        self._cosmo = Cosmology()

    def _get_N0_phi(self, ellmax):
        return np.concatenate((np.zeros(self.cmb_offset), self.N0[0][:ellmax + 1 - self.cmb_offset]))

    def _get_N0_curl(self, ellmax):
        return np.concatenate((np.zeros(self.cmb_offset), self.N0[1][:ellmax + 1 - self.cmb_offset]))

    def _get_N0_kappa(self, ellmax):
        ells = np.arange(ellmax + 1)
        return self._get_N0_phi(ellmax) * 0.25 * (ells + 0.5) ** 4

    def _get_N0_omega(self, ellmax):
        ells = np.arange(ellmax + 1)
        return self._get_N0_curl(ellmax) * 0.25 * (ells + 0.5) ** 4

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > self.cmb_offset + 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def setup_cmb_noise(self, cmb_noise_file, cmb_offset):
        self.N0 = np.load(cmb_noise_file)
        self.cmb_offset = cmb_offset

    def get_N0(self, typ="phi", ellmax=4000, tidy=False, ell_factors=False):
        """
        Extracts the noise from the supplied input file.

        Parameters
        ----------
        typ : str
            'phi' or 'curl'.
        ellmax : int
            The maximum multipole moment to return.
        tidy : bool
            Whether to interpole values less than zero.
        ell_factors : bool
            Whether to multiply the noise by (1/4)(ell + 1/2)^4

        Returns
        -------
        ndarray
            1D array of the noise up to desired ellmax, the indices representing ell - offset.
        """
        if self.N0 is None:
            raise ValueError(f"N0 has not been created. Please run 'setup_cmb_noise(self, cmb_noise_file, cmb_offset)' first.")
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

    def get_gal_shot_N(self, n=40, ellmax=4000):
        arcmin2_to_strad = 11818080
        ones = np.ones(ellmax + 1)
        return ones/(arcmin2_to_strad * n)

    def get_cib_shot_N(self, nu, ellmax=4000):
        # 1309.0382 Table 9
        ones = np.ones(ellmax + 1)
        if nu == 353e9:
            N = 262 * 1e-12    # 1e-12 to change units to MJy^2/sr
        elif nu == 545e9:
            N = 1690 * 1e-12
        elif nu == 857e9:
            N = 5364 * 1e-12
        return ones * N

    def _microK2_to_MJy2(self, value, nu):
        if nu == 353e9:
            factor = 287
        elif nu == 545e9:
            factor = 58
        elif nu == 857e9:
            factor = 2.3
        return value * 1e-12 * factor ** 2

    def get_dust_N(self, nu, ellmax=4000):
        # 1303.5075 eq 9,
        ells = np.arange(ellmax + 1)
        # alpha = 0.387             # parameters from 1609.08942 pg 8
        # l_c = 162.9
        # gamma = 0.168
        alpha = 0.169              # parameters from 1303.5075 pg 6
        l_c = 905
        gamma = 0.427
        if nu == 353e9:
            A = 6e2              # Tuned to match 1705.02332 fig 2 and 3, and 2110.09730 fig 2
        elif nu == 545e9:
            A = 2e5              # Complete guess
        elif nu == 857e9:
            A = 6e8         # from 1303.5075 pg 6
        D_l = self._microK2_to_MJy2(A, nu) * ((100 / ells) ** alpha / ((1 + (ells / l_c) ** 2) ** (gamma / 2)))
        return 2*np.pi * D_l/(ells*(ells+1))

    def get_cmb_N(self, nu, ellmax=4000):
        # Important for 353e9 according to 1609.08942
        cmb_ps = self._cosmo.get_cmb_ps(ellmax)
        return self._microK2_to_MJy2(cmb_ps, nu)
