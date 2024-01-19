import numpy as np

import omegaqe
from omegaqe.cosmology import Cosmology
from omegaqe.tools import getFileSep
import pandas as pd


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

    def __init__(self, cosmology=None, full_sky=False):
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
        self.cmb_offset = 2
        self.cosmo = Cosmology() if cosmology is None else cosmology
        self.full_sky = full_sky
        self.agora = self.cosmo.agora

    def _get_N0_phi(self, ellmax):
        return np.concatenate((np.zeros(self.cmb_offset), self.N0[0][:ellmax + 1 - self.cmb_offset]))

    def _get_N0_curl(self, ellmax):
        return np.concatenate((np.zeros(self.cmb_offset), self.N0[1][:ellmax + 1 - self.cmb_offset]))

    def _get_N0_kappa(self, ellmax):
        ells = np.arange(ellmax + 1)
        fac = 0.25 * (ells * (ells+1)) ** 2 if self.full_sky else 0.25 * (ells) ** 4
        return self._get_N0_phi(ellmax) * fac

    def _get_N0_omega(self, ellmax):
        ells = np.arange(ellmax + 1)
        fac = 0.25 * (ells * (ells+1)) ** 2 if self.full_sky else 0.25 * (ells) ** 4
        return self._get_N0_curl(ellmax) * fac

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > self.cmb_offset + 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def _get_N0(self, exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, iter_ext, data_dir):
        print(f"Getting cached N0 for exp: {exp}, qe: {qe}, gmv: {gmv}, ps: {ps}, L_cuts: {(T_Lmin, T_Lmax, P_Lmin, P_Lmax)}, iter: {iter}, iter_ext: {iter_ext}, data_dir: {data_dir}")
        if iter_ext:
            qe += "_iter_ext"
        elif iter:
            qe += "_iter"
        elif gmv:
            qe += "_gmv"
        sep = getFileSep()
        dir = f'{data_dir}{sep}N0{sep}{exp}{sep}'
        N0_phi = np.array(pd.read_csv(dir+f'N0_phi_{ps}_T{T_Lmin}-{T_Lmax}_P{P_Lmin}-{P_Lmax}.csv', sep=' ')[qe])
        N0_curl = np.array(pd.read_csv(dir+f'N0_curl_{ps}_T{T_Lmin}-{T_Lmax}_P{P_Lmin}-{P_Lmax}.csv', sep=' ')[qe])
        return N0_phi, N0_curl

    def setup_cmb_noise(self, exp="SO", qe="TEB", gmv=True, ps="gradient", T_Lmin=30, T_Lmax=3000, P_Lmin=30, P_Lmax=5000, iter=False, iter_ext=False, data_dir=omegaqe.DATA_DIR):
        print(f"Setting up noise...")
        self.N0 = self._get_N0(exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, iter_ext, data_dir)

    def get_N0(self, typ, ellmax, exp="SO", qe="TEB", gmv=True, ps="gradient", T_Lmin=30, T_Lmax=3000, P_Lmin=30, P_Lmax=5000, recalc_N0=False, iter=False, iter_ext=False, data_dir=omegaqe.DATA_DIR):
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
        if recalc_N0:
            self.N0 = self._get_N0(exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, iter_ext, data_dir)
        if self.N0 is None:
            raise ValueError(f"N0 has not been created, either call setup_cmb_noise or use recalc_N0 argument.")
        if typ == "phi":
            return self._get_N0_phi(ellmax)
        if typ == "curl":
            return self._get_N0_curl(ellmax)
        if typ == "kappa":
            return self._get_N0_kappa(ellmax)
        if typ == "omega":
            return self._get_N0_omega(ellmax)
        raise ValueError(f"Supplied type {typ} not in [phi, kappa, curl, omega].")

    def get_gal_shot_N(self, n=40, ellmax=4000, zmin=None, zmax=None):
        fraction = 1
        if zmin is not None and zmax is not None:
            fraction = self.cosmo.gal_window_fraction(zmin, zmax)
        arcmin2_to_strad = 11818080
        ones = np.ones(ellmax + 1)
        return ones/(arcmin2_to_strad * n * fraction)

    def get_shape_N(self, n=40, sig=0.21, ellmax=4000, zmin=None, zmax=None):
        return self.get_gal_shot_N(n, ellmax, zmin, zmax) * sig**2

    def get_cib_shot_N(self, nu, ellmax=4000):
        # 1309.0382 Table 9
        ones = np.ones(ellmax + 1)
        print(self.agora)
        if nu == 353e9:
            if self.agora: return 426e-12  # My fit of AGORA cib between ell of 110 and 2000
            # N = 262 * 1e-12    # 1e-12 to change units to MJy^2/sr
            N = 225.6 * 1e-12   # From Toshiya, matching 1705.02332 and 2110.09730
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
        # From Toshiya, matching 1705.02332 and 2110.09730
        ells = np.arange(ellmax + 1)
        alpha = 2.17
        if nu == 353e9:
            A = 0.00029989393
        else:
            return self.get_dust_N_old(nu, ellmax)
        return A * ells**(-alpha)

    def get_dust_N_old(self, nu, ellmax=4000):
        # 1303.5075 eq 9,
        ells = np.arange(ellmax + 1)
        # alpha = 0.387             # parameters from 1609.08942 pg 8
        # l_c = 162.9
        # gamma = 0.168

        # return 0.00029989393*ells**(-2.17)

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
        N_dust = 2*np.pi * D_l/(ells*(ells+1))
        N_dust[0] = 0
        return N_dust

    def get_dust_N_fit(self, nu, alpha, l_c, gamma, A, ellmax=4000):
        # 1303.5075 eq 9,
        ells = np.arange(ellmax + 1)
        D_l = self._microK2_to_MJy2(A, nu) * ((100 / ells) ** alpha / ((1 + (ells / l_c) ** 2) ** (gamma / 2)))
        N_dust = 2*np.pi * D_l/(ells*(ells+1))
        N_dust[0] = 0
        return N_dust

    def get_noise_args(self, exp):
        if exp == "SO":
            return 3, 3
        elif exp == "SO_base":
            return None, None
        elif exp == "SO_goal":
            return None, None
        elif exp == "S4":
            return 1, 3
        elif exp == "S4_dp":
            return 0.4, 2.3
        elif exp == "S4_test":
            return 1, 1
        elif exp == "S4_base":
            return None, None
        elif exp == "HD":
            return 0.5, 0.25
        else:
            raise ValueError(f"Experiment {exp} unexpected.")

    def get_cmb_N(self, nu, ellmax=4000):
        # Important for 353e9 according to 1609.08942
        cmb_ps = self.cosmo.get_cmb_ps(ellmax)
        return self._microK2_to_MJy2(cmb_ps, nu)

    def _get_cached_cmb_gaussian_N(self, typ, ellmax, exp, data_dir):
        if typ[0] != typ[1]:
            return np.zeros(ellmax + 1)
        sep = getFileSep()
        dir = f'{data_dir}{sep}N0{sep}{exp}{sep}'
        N = np.array(pd.read_csv(dir + f'N.csv', sep=' ')[typ[0]])[:ellmax + 1]
        N = np.concatenate((np.zeros(self.cmb_offset), N))
        return N

    def get_cmb_gaussian_N(self, typ, deltaT=3, beam=3, ellmax=4000, exp="SO", data_dir=omegaqe.DATA_DIR):
        """

        Parameters
        ----------
        typ
        deltaT
        beam
        ellmax

        Returns
        -------

        """
        if deltaT is None or beam is None:
            return self._get_cached_cmb_gaussian_N(typ, ellmax, exp, data_dir)

        arcmin_to_radians = 0.000290888   #pi/180/60
        deltaT *= arcmin_to_radians
        beam *= arcmin_to_radians
        T_cmb = 2.7255               #arXiv:0911.1955
        Ls = np.arange(ellmax+1)
        if typ == "TT":
            return (deltaT*1e-6/T_cmb)**2 * np.exp(Ls*(Ls+1)*beam**2/(8*np.log(2)))
        elif typ == "EE" or typ == "BB":
            return (deltaT * 1e-6 *np.sqrt(2)/ T_cmb) ** 2 * np.exp(Ls * (Ls + 1) * beam**2 / (8 * np.log(2)))
        else:
            return np.zeros(np.size(Ls))
