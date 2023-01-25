import numpy as np
import copy
from powerspectra import Powerspectra
from noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
from sympy.matrices import Matrix
from sympy import lambdify

class Covariance:


    def __init__(self):
        self.noise = Noise()
        self.power = Powerspectra()
        self.binned_gal_types = list("abcdef")
        self.test_types = list("xyz")

    def get_log_sample_Ls(self, Lmin, Lmax, Nells=500, dL_small=1):
        floaty = Lmax / 1000
        samp1 = np.arange(Lmin, floaty * 10, dL_small)
        samp2 = np.logspace(1, 3, Nells-np.size(samp1)) * floaty
        return np.concatenate((samp1, samp2))

    def setup_cmb_noise(self, exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, data_dir):
        self.noise.setup_cmb_noise(exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, data_dir)

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample[1:], arr[1:])

    def _get_Cl_kappa(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_kappa_ps(ells)

    def _get_Cl_gal(self,ellmax, gal_win_zmin_a=None, gal_win_zmax_a=None, gal_win_zmin_b=None, gal_win_zmax_b=None, use_bins=True, gal_distro="LSST_gold"):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_gal_ps(ells, gal_win_zmin_a=gal_win_zmin_a, gal_win_zmax_a=gal_win_zmax_a, gal_win_zmin_b=gal_win_zmin_b, gal_win_zmax_b=gal_win_zmax_b, gal_distro=gal_distro)
        return self.power.get_gal_ps(ells, gal_distro=gal_distro)

    def _get_Cl_cib(self,ellmax, nu=353e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_ps(ells, nu=nu)

    def _get_Cl_gal_kappa(self,ellmax, gal_win_zmin=None, gal_win_zmax=None, use_bins=True, gal_distro="LSST_gold"):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_gal_kappa_ps(ells, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, gal_distro=gal_distro)
        return self.power.get_gal_kappa_ps(ells, gal_distro=gal_distro)

    def _get_Cl_cib_kappa(self,ellmax, nu=353e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_kappa_ps(ells, nu=nu)

    def _get_Cl_cib_gal(self,ellmax, nu=353e9, gal_win_zmin=None, gal_win_zmax=None, use_bins=True, gal_distro="LSST_gold"):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_cib_gal_ps(ells, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, gal_distro=gal_distro)
        return self.power.get_cib_gal_ps(ells, nu=nu, gal_distro=gal_distro)

    def _get_Cl(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold"):
        if "g" not in typ:
            return self.power.get_ps(typ, np.arange(ellmax + 1), nu=nu)
        if typ == "kk":
            return self._get_Cl_kappa(ellmax)
        elif typ == "gk" or typ == "kg":
            return self._get_Cl_gal_kappa(ellmax, gal_bins[0], gal_bins[1], use_bins, gal_distro=gal_distro)
        elif typ == "gg":
            return self._get_Cl_gal(ellmax, gal_bins[0], gal_bins[1], gal_bins[2], gal_bins[3], use_bins, gal_distro=gal_distro)
        elif typ == "Ik" or typ == "kI":
            return self._get_Cl_cib_kappa(ellmax, nu)
        elif typ == "II":
            return self._get_Cl_cib(ellmax, nu)
        elif typ == "Ig" or typ == "gI":
            return self._get_Cl_cib_gal(ellmax, nu, gal_bins[0], gal_bins[1], use_bins, gal_distro=gal_distro)
        elif typ == "ww":
            N0_omega = self.noise.get_N0("omega", ellmax)
            return N0_omega

        gal_win_zmin_1 = None
        gal_win_zmax_1 = None
        gal_win_zmin_2 = None
        gal_win_zmax_2 = None
        type_0_binned = False
        if typ[0] in self.binned_gal_types:
            index_1 = 2 * (ord(typ[0]) - ord("a"))
            gal_win_zmin_1 = gal_bins[index_1]
            gal_win_zmax_1 = gal_bins[index_1 + 1]
            typ = "g" + typ[1]
            type_0_binned = True
        if typ[1] in self.binned_gal_types:
            index_2 = 2*(ord(typ[1]) - ord("a"))
            typ = typ[0] + "g"
            if type_0_binned:
                gal_win_zmin_2 = gal_bins[index_2]
                gal_win_zmax_2 = gal_bins[index_2 + 1]
            else:
                gal_win_zmin_1 = gal_bins[index_2]
                gal_win_zmax_1 = gal_bins[index_2 + 1]
        return self._get_Cl(typ, ellmax, nu, (gal_win_zmin_1, gal_win_zmax_1, gal_win_zmin_2, gal_win_zmax_2), use_bins=True, gal_distro=gal_distro)

    def _get_Cov(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold"):
        if typ[0] != typ[1]:
            if typ[0] in self.test_types and typ[1] in self.test_types:
                return self._get_Cl_kappa(ellmax)
            return self._get_Cl(typ, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro)
        if typ[0] == "k":
            N = self.noise.get_N0("kappa", ellmax)
        elif typ[0] in self.test_types:
            print("test")
            N = 2*self.noise.get_N0("kappa", ellmax)
            typ="kk"
        elif typ[0] == "I":
            N_cib = self.noise.get_cib_shot_N(ellmax=ellmax, nu=nu)
            N_dust = self.noise.get_dust_N(ellmax=ellmax, nu=nu)
            N = N_cib + N_dust
            N[:110] = 1e10
            N[2001:] = 1e10
        elif typ[0] == "s":
            N = self.noise.get_shape_N(ellmax=ellmax)
        elif typ[0] == "g":
            N = self.noise.get_gal_shot_N(ellmax=ellmax)
        elif typ[0] in self.binned_gal_types:
            index = 2 * (ord(typ[0]) - ord("a"))
            gal_win_zmin = gal_bins[index]
            gal_win_zmax = gal_bins[index + 1]
            N = self.noise.get_gal_shot_N(ellmax=ellmax, zmin=gal_win_zmin, zmax=gal_win_zmax)
            # N = 1e-100
            return self._get_Cl_gal(ellmax, gal_win_zmin, gal_win_zmax, gal_win_zmin, gal_win_zmax, gal_distro=gal_distro) + N
        else:
            raise ValueError(f"Could not get Cov for type {typ}")
        return self._get_Cl(typ, ellmax, nu, gal_bins, gal_distro=gal_distro) + N

    def _get_C_inv(self, typs, Lmax, nu, gal_bins, gal_distro="LSST_gold"):
        Ntyps = np.size(typs)
        typs_no_fI = copy.deepcopy(typs)
        typs_no_fI[typs_no_fI == "f"] = "z"        # Replacing 'f' with 'z' for sympy operations as 'ff' is sympy function
        typs_no_fI[typs_no_fI == "I"] = "y"
        C = typs[:, None] + typs[None, :]
        C_no_fI = typs_no_fI[:, None] + typs_no_fI[None, :]
        args = C.flatten()
        args_no_fI = C_no_fI.flatten()
        C_sym = Matrix(C_no_fI)
        if Ntyps > 3:
            C_inv = C_sym.inv('LU')
        else:
            C_inv = C_sym.inv()
        C_inv_func = lambdify(args_no_fI, C_inv)
        Covs = [self._get_Cov(arg, Lmax, nu, gal_bins, gal_distro=gal_distro) for arg in args]
        return C_inv_func(*Covs)

    def get_C_inv(self, typs, Lmax, nu, gal_bins=(None, None, None, None), gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        typs
        Lmax
        nu
        gal_bins

        Returns
        -------

        """
        return self._get_C_inv(np.char.array(list(typs)), Lmax, nu, gal_bins, gal_distro)

    def get_Cov(self, typ, ellmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        typ
        ellmax
        nu
        gal_bins
        use_bins

        Returns
        -------

        """
        return self._get_Cov(typ, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro)

    def get_Cl(self, typ, ellmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        typ
        ellmax
        nu
        gal_bins
        use_bins

        Returns
        -------

        """
        return self._get_Cl(typ, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro)
