import numpy as np
import copy
from omegaqe.powerspectra import Powerspectra
from omegaqe.noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
from sympy.matrices import Matrix
from sympy import lambdify

class Covariance:


    def __init__(self, cosmology=None):
        self.noise = Noise(cosmology=cosmology)
        self.power = Powerspectra(cosmology=self.noise.cosmo)
        self.binned_gal_types = list("abcdef")
        self.test_types = list("xyz")
        self.use_LSST_abcde = False
        self.shot_noise = [2.25, 3.11, 3.09, 2.61, 2.00] if self.use_LSST_abcde else None  # caution that one bin agora is using n=40 

    def get_log_sample_Ls(self, Lmin, Lmax, Nells=500, dL_small=1):
        floaty = Lmax / 1000
        samp1 = np.arange(Lmin, floaty * 10, dL_small)
        samp2 = np.logspace(1, 3, Nells-np.size(samp1)) * floaty
        return np.concatenate((samp1, samp2))

    def setup_cmb_noise(self, exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, iter_ext, data_dir):
        self.noise.setup_cmb_noise(exp, qe, gmv, ps, T_Lmin, T_Lmax, P_Lmin, P_Lmax, iter, iter_ext, data_dir)

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample[1:], arr[1:])

    def _get_Cl_kappa(self, ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_kappa_ps(ells)

    def _get_Cl_gal(self, ellmax, gal_win_zmin_a=None, gal_win_zmax_a=None, gal_win_zmin_b=None, gal_win_zmax_b=None, use_bins=True, gal_distro="LSST_gold", gal_distro_b=None):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_gal_ps(ells, gal_win_zmin_a=gal_win_zmin_a, gal_win_zmax_a=gal_win_zmax_a, gal_win_zmin_b=gal_win_zmin_b, gal_win_zmax_b=gal_win_zmax_b, gal_distro=gal_distro, gal_distro_b=gal_distro_b)
        return self.power.get_gal_ps(ells, gal_distro=gal_distro, gal_distro_b=gal_distro_b)

    def _get_Cl_cib(self, ellmax, nu=353e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_ps(ells, nu=nu)

    def _get_Cl_gal_kappa(self, ellmax, gal_win_zmin=None, gal_win_zmax=None, use_bins=True, gal_distro="LSST_gold"):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_gal_kappa_ps(ells, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, gal_distro=gal_distro)
        return self.power.get_gal_kappa_ps(ells, gal_distro=gal_distro)

    def _get_Cl_cib_kappa(self, ellmax, nu=353e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_kappa_ps(ells, nu=nu)

    def _get_Cl_cib_gal(self, ellmax, nu=353e9, gal_win_zmin=None, gal_win_zmax=None, use_bins=True, gal_distro="LSST_gold"):
        ells = np.arange(ellmax + 1)
        if use_bins:
            return self.power.get_cib_gal_ps(ells, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, gal_distro=gal_distro)
        return self.power.get_cib_gal_ps(ells, nu=nu, gal_distro=gal_distro)

    def _get_Cl(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold", gal_distro_b=None):
        if "s" in typ:
            return self.power.get_ps(typ, np.arange(ellmax + 1), nu=nu)
        if typ == "kk":
            return self._get_Cl_kappa(ellmax)
        elif typ == "gk" or typ == "kg":
            return self._get_Cl_gal_kappa(ellmax, gal_bins[0], gal_bins[1], use_bins, gal_distro=gal_distro)
        elif typ == "gg":
            return self._get_Cl_gal(ellmax, gal_bins[0], gal_bins[1], gal_bins[2], gal_bins[3], use_bins, gal_distro=gal_distro, gal_distro_b=gal_distro_b)
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
        gal_distro_b = None
        if typ[0] in self.binned_gal_types:
            if self.use_LSST_abcde:
                gal_distro = f"LSST_{typ[0]}"
            else:
                index_1 = 2 * (ord(typ[0]) - ord("a"))
                gal_win_zmin_1 = gal_bins[index_1]
                gal_win_zmax_1 = gal_bins[index_1 + 1]
                type_0_binned = True
            typ = "g" + typ[1]
        if typ[1] in self.binned_gal_types:
            if self.use_LSST_abcde:
                gal_distro_b = f"LSST_{typ[1]}"
            else:
                index_2 = 2*(ord(typ[1]) - ord("a"))
                if type_0_binned:
                    gal_win_zmin_2 = gal_bins[index_2]
                    gal_win_zmax_2 = gal_bins[index_2 + 1]
                else:
                    gal_win_zmin_1 = gal_bins[index_2]
                    gal_win_zmax_1 = gal_bins[index_2 + 1]
            typ = typ[0] + "g"
        return self._get_Cl(typ, ellmax, nu, (gal_win_zmin_1, gal_win_zmax_1, gal_win_zmin_2, gal_win_zmax_2), use_bins=True, gal_distro=gal_distro, gal_distro_b=gal_distro_b)

    def _get_Cov(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold", noise=True):
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
            if self.use_LSST_abcde:
                N = self.noise.get_gal_shot_N(ellmax=ellmax, n=self._get_n(typ[0]))
                gal_distro = f"LSST_{typ[0]}"
                return self._get_Cl_gal(ellmax, gal_distro=gal_distro) + N
            gal_win_zmin = gal_bins[index]
            gal_win_zmax = gal_bins[index + 1]
            N = self.noise.get_gal_shot_N(ellmax=ellmax, zmin=gal_win_zmin, zmax=gal_win_zmax)
            # N = 1e-100
            cl = self._get_Cl_gal(ellmax, gal_win_zmin, gal_win_zmax, gal_win_zmin, gal_win_zmax, gal_distro=gal_distro)
            if noise:
                return cl + N
            return cl
        else:
            raise ValueError(f"Could not get Cov for type {typ}")
        cl = self._get_Cl(typ, ellmax, nu, gal_bins, gal_distro=gal_distro)
        if noise:
            return cl + N
        return cl
    
    def _get_Cov_mat(self, typs, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold", noise=True):
        typs = np.char.array(list(typs))
        Ntyps = np.size(typs)
        cov_mat = np.empty((Ntyps, Ntyps, ellmax+1))
        for iii in np.arange(Ntyps):
            for jjj in np.arange(iii, Ntyps):
                cov = self._get_Cov(typs[iii]+typs[jjj], ellmax, nu, gal_bins, use_bins, gal_distro, noise)
                cov_mat[iii, jjj, :] = cov_mat[jjj, iii, :] = cov 
        return cov_mat


    def _get_n(self, typ):
        if self.shot_noise is None:
            return 40
        idx = ord(typ[0]) - ord("a")
        return self.shot_noise[idx]

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

    def _get_rho(self, typ, Lmax, nu, gal_bins, use_bins, gal_distro, include_kappa_noise=True):
        a_1 = typ[0]
        a_2 = typ[1]
        typ1 = a_1 + a_1
        typ2 = a_2 + a_2
        if typ1 == "kk" and not include_kappa_noise:
            cov_a1 = self._get_Cl(typ1, Lmax, nu, gal_bins, use_bins, gal_distro)
        else:
            cov_a1 = self._get_Cov(typ1, Lmax, nu, gal_bins, use_bins, gal_distro)
        if typ2 == "kk" and not include_kappa_noise:
            cov_a2 = self._get_Cl(typ2, Lmax, nu, gal_bins, use_bins, gal_distro)
        else:
            cov_a2 = self._get_Cov(typ2, Lmax, nu, gal_bins, use_bins, gal_distro)
        if a_1 + a_2 == "kk" and not include_kappa_noise:
            cov_cross = self._get_Cl(a_1 + a_2, Lmax, nu, gal_bins, use_bins, gal_distro)
        else:
            cov_cross = self._get_Cov(a_1 + a_2, Lmax, nu, gal_bins, use_bins, gal_distro)
        corr = cov_cross / (np.sqrt(cov_a1 * cov_a2))
        return corr

    def _get_corr_inv(self, typs, Lmax, nu, gal_bins, use_bins, gal_distro):
        Ntyps = np.size(typs)
        typs_no_fI = copy.deepcopy(typs)
        typs_no_fI[typs_no_fI == "f"] = "z"  # Replacing 'f' with 'z' for sympy operations as 'ff' is sympy function
        typs_no_fI[typs_no_fI == "I"] = "y"
        rho = typs[:, None] + typs[None, :]
        rho_no_fI = typs_no_fI[:, None] + typs_no_fI[None, :]
        args = rho.flatten()
        args_no_fI = rho_no_fI.flatten()
        rho_sym = Matrix(rho_no_fI)
        if Ntyps > 3:
            rho_inv = rho_sym.inv('LU')
        else:
            rho_inv = rho_sym.inv()
        rho_inv_func = lambdify(args_no_fI, rho_inv)
        rhos = [self._get_rho(arg, Lmax, nu, gal_bins, use_bins, gal_distro) for arg in args]
        return rho_inv_func(*rhos)

    def _get_total_tracer_corr(self, tracers, Lmax, nu, gal_bins, use_bins, gal_distro):
        rho_inv = self._get_corr_inv(tracers, Lmax, nu, gal_bins, use_bins, gal_distro)
        rho = np.zeros(Lmax + 1)
        for iii, a_i in enumerate(tracers):
            rho_ik = self._get_rho(a_i + "k", Lmax, nu, gal_bins, use_bins, gal_distro, include_kappa_noise=False)
            for jjj, a_j in enumerate(tracers):
                rho_jk = self._get_rho(a_j + "k", Lmax, nu, gal_bins, use_bins, gal_distro, include_kappa_noise=False)
                rho += rho_ik * rho_inv[iii][jjj] * rho_jk
        return np.sqrt(rho)
    
    def _get_delens_rho(self, Lmax, zerod=True):
        ells = np.arange(Lmax + 1)
        cl_kappa = self.power.get_kappa_ps(ells, use_weyl=False)
        cl_phi = 4/(ells*(ells+1))**2 * cl_kappa
        cl_phi[0] = 0
        N0 = self.noise.get_N0("phi", Lmax)
        rho = cl_phi/(cl_phi + N0) 
        if zerod:
            rho[0] = 0
        return np.sqrt(rho)

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

    def get_corr_inv(self, typs, Lmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold"):
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
        return self._get_corr_inv(np.char.array(list(typs)), Lmax, nu, gal_bins, use_bins, gal_distro)

    def get_total_tracer_corr(self, tracers, Lmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        tracers
        Lmax
        nu
        gal_bins
        use_bins
        gal_distro

        Returns
        -------

        """
        return self._get_total_tracer_corr(np.char.array(list(tracers)), Lmax, nu, gal_bins, use_bins, gal_distro)

    def get_corr(self, typ, Lmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold", include_kappa_noise=True):
        """

        Parameters
        ----------
        typ
        Lmax
        nu
        gal_bins
        use_bins
        gal_distro

        Returns
        -------

        """
        return self._get_rho(typ, Lmax, nu, gal_bins, use_bins, gal_distro, include_kappa_noise)
    
    def get_delens_corr(self, Lmax):
        """

        Parameters
        ----------
        typ
        Lmax
        nu
        gal_bins
        use_bins
        gal_distro

        Returns
        -------

        """
        return self._get_delens_rho(Lmax)

    def get_Cov(self, typ, ellmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold", noise=True):
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
        return self._get_Cov(typ, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro, noise=noise)
    
    def get_Cov_mat(self, typs, ellmax, nu=353e9, gal_bins=(None, None, None, None), use_bins=False, gal_distro="LSST_gold", noise=True):
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
        return self._get_Cov_mat(typs, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro, noise=noise)

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
