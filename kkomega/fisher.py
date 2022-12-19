import numpy as np
from bispectra import Bispectra
from covariance import Covariance
from postborn import omega_ps
from scipy.interpolate import InterpolatedUnivariateSpline
from cache.tools import getFileSep, path_exists
import copy
import warnings
import vector
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Fisher:
    """
    Calculates the Fisher information of post born lensing bispectra.

    Attributes
    ----------
    bi : Bispetrum
        Instance of Bispectrum, this instantiation will spline the supplied modecoupling matrix.
    power : Powerspectra
    """

    def __init__(self, exp="SO", qe="TEB", gmv=True, ps="gradient", L_cuts=(30,3000,30,5000), iter=False, data_dir="data", setup_bispectra=False):
        """
        Constructor

        Parameters
        ----------

        """
        self.covariance = Covariance()
        self.setup_noise(exp, qe, gmv, ps, L_cuts, iter, data_dir)
        self.bi = Bispectra()
        if setup_bispectra:
            self.setup_bispectra()
        self.power = self.covariance.power
        self.opt_I_cache = None
        self.C_inv = None
        self.C_omega_spline = None

    def _files_match(self, ell_file, M_file):
        if ell_file[:-8] == M_file[:-5]:
            return True
        return False

    def _check_path(self, path):
        if not path_exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, data_dir=None):
        if exp is not None: self.exp = exp
        if qe is not None: self.qe = qe
        if gmv is not None: self.gmv = gmv
        if ps is not None: self.ps = ps
        if L_cuts is not None: self.L_cuts = L_cuts
        if iter is not None: self.iter = iter
        if data_dir is not None: self.data_dir = data_dir
        self.reset_noise()

    def setup_bispectra(self, path="cache/_M", ellmax=5000, Nell=100):
        """

        Parameters
        ----------
        path
        ellmax
        Nell

        Returns
        -------

        """
        self.mode_path = path
        M_types = ["kk", "gg", "gk", "kg", "II", "Ik", "kI", "gI", "Ig"]
        # M_types = Modecoupling().get_M_types()
        M_dir = f"{ellmax}_{Nell}_s"
        sep = getFileSep()
        for M_type in M_types:
            dir = path+sep+M_type
            self._check_path(dir)
            dir += sep+M_dir+sep
            self._check_path(dir)
            ells = np.load(f"{dir}ells.npy")
            M = np.load(f"{dir}M.npy")
            self.bi.build_M_spline(M_type, ells, M)

    def _get_third_L(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 + (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample[1:], arr[1:])

    def _get_Covs(self, typ, Lmax, all_splines=False, nu=353e9, gal_bins=(None,None,None,None), include_N0_kappa="both", gal_distro="LSST_gold"):
        N0_omega_spline = self._interpolate(self.covariance.noise.get_N0("omega", Lmax))
        C3_spline = N0_omega_spline
        if typ == "kkw":
            if include_N0_kappa == "both":
                C1 = self.covariance.get_Cov("kk", Lmax)
                C2 = copy.deepcopy(C1)
            elif include_N0_kappa == "one":
                N0_kappa = self.covariance.noise.get_N0("kappa", Lmax)
                Cl_kappa = self.covariance.get_Cl("kk", Lmax)
                C1 = Cl_kappa + (0.5*N0_kappa)
                C2 = Cl_kappa
            else:
                Cl_kappa = self.covariance.get_Cl("kk", Lmax)
                C1 = Cl_kappa
                C2 = Cl_kappa
        else:
            typ1 = typ[0]
            typ2 = typ[1]
            C1 = self.covariance.get_Cov(typ1 + typ1, Lmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            C2 = self.covariance.get_Cov(typ2 + typ2, Lmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

    def _get_optimal_Ns_sympy(self, Lmax, typ, typs, C_inv, all_spline=False, return_cov3=True):
        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        if all_spline:
            cov_inv1 = self._interpolate(cov_inv1)
            cov_inv2 = self._interpolate(cov_inv2)
        if not return_cov3:
            return cov_inv1, cov_inv2
        N0_omega_spline = self._interpolate(self.covariance.noise.get_N0("omega", Lmax))
        cov3 = N0_omega_spline
        return cov_inv1, cov_inv2, cov3

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ, Lmin, nu, gal_bins, typs=None, C_inv=None, include_N0_kappa="both", gal_distro="LSST_gold"):
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(Lmin, Lmax + 1, dL)
        L3 = self._get_third_L(Ls[:, None], Ls[None, :], thetas[:, None, None])
        w = np.ones(np.shape(L3))
        w[L3 < Lmin] = 0
        w[L3 > Lmax] = 0
        if typ[:3] == "opt":
            C1, C2, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv)
            denom = C1[None, Ls, None] * C2[None, None, Ls] / C3_spline(L3)             # These are actually the C_inv
        else:
            C1, C2, C3_spline = self._get_Covs(typ, Lmax, all_splines=False, nu=nu, gal_bins=gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
            if typ[0] != typ[1]:
                Cl = self.covariance.get_Cl(typ[:2], Lmax, nu, gal_bins, gal_distro=gal_distro)
                denom = ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl[None, Ls, None] * Cl[None, None, Ls])) * C3_spline(L3)
            else:
                denom = 2 * C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)
        return Ls, L3, dTheta, w, denom

    def _get_bispectrum_Fisher_vec(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa, Lmin, nu, gal_bins, gal_distro="LSST_gold"):
        Ls, L3, dTheta, w, denom = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin, nu, gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        bispectrum = self.bi.get_bispectrum(typ, Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bispectrum ** 2 / denom)
        return f_sky/np.pi * I/((2*np.pi)**2)

    def _integral_prep_sample(self, Ls, Ntheta, typ, nu, gal_bins, typs=None, C_inv=None, include_N0_kappa="both", gal_distro="LSST_gold"):
        Lmax = int(np.max(Ls))
        Lmin = int(np.min(Ls))
        if typ[:3] == "opt":
            C1_spline, C2_spline, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv, all_spline=True)
        else:
            C1_spline, C2_spline, C3_spline = self._get_Covs(typ, Lmax, all_splines=True, nu=nu, gal_bins=gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[:-1] = Ls[1:] - Ls[0:-1]
        dLs[-1] = dLs[-2]
        return Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    def _integral_prep_F_L(self, Lmax, Ntheta, typ, typs, C_inv):
        C1_spline, C2_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv, all_spline=True, return_cov3=False)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        return thetas, dTheta, weights, C1_spline, C2_spline

    def _get_bispectrum_Fisher_sample(self, typ, Ls, dL2, Ntheta, f_sky, arr, include_N0_kappa, nu, gal_bins, gal_distro="LSST_gold"):
        Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta,typ, nu, gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        Cl_xy_spline = self._interpolate(self.covariance.get_Cl(typ[:2], Lmax, nu, gal_bins, gal_distro=gal_distro))
        I = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax+1, dL2)
        for iii, L3 in enumerate(Ls):
            I_tmp = 0
            L2 = Ls2[None, :]
            L3_vec = vector.obj(rho=L3, phi=0)
            L2_vec = vector.obj(rho=L2, phi=thetas[:, None])
            L1_vec = -L3_vec - L2_vec
            L1 = L1_vec.rho
            w = np.ones(np.shape(L1))
            w[L1 > Lmax] = 0
            w[L1 < Lmin] = 0
            thetas12 = L1_vec.deltaphi(L2_vec)
            bispectrum = self.bi.get_bispectrum(typ, L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            if typ[0] == typ[1]:
                denom = 2 * C1_spline(L1) * C2_spline(L2) * C3_spline(L3)
            else:
                denom = ((C1_spline(L1) * C2_spline(L2)) + (Cl_xy_spline(L1) * Cl_xy_spline(L2))) * C3_spline(L3)
            I_tmp += dL2 * 2 * np.sum(L2 * w * dTheta * bispectrum ** 2 / denom)
            I[iii] = 2 * np.pi * L3 * I_tmp
        I *= f_sky / np.pi * 1 / ((2 * np.pi) ** 2)
        if arr:
            return I
        I_spline = InterpolatedUnivariateSpline(Ls, I)
        return I_spline.integral(Lmin, Lmax)

    def _get_optimal_bispectrum_Fisher_element_vec(self, typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_bins, gal_distro="LSST_gold"):
        Ls, L3, dTheta, w, covs = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin=Lmin, nu=nu, gal_bins=gal_bins, typs=typs, C_inv=C_inv, gal_distro=gal_distro)
        bi1 = self.bi.get_bispectrum(typ[4:6] + "w", Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        bi2 = self.bi.get_bispectrum(typ[6:] + "w", Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bi1 * bi2 * covs)
        return 0.5 * f_sky/np.pi * I / ((2 * np.pi) ** 2)

    def _get_optimal_bispectrum_Fisher_sample(self, typs, typ, Ls, dL2, Ntheta, f_sky, C_inv, nu, gal_bins, save_array, gal_distro="LSST_gold"):
        Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta,typ, nu, gal_bins, typs=typs, C_inv=C_inv, gal_distro=gal_distro)
        if save_array and self.opt_I_cache is None:
            self.opt_I_cache = np.zeros(np.size(Ls))
            self.opt_Ls = Ls
        I = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax + 1, dL2)
        if any([np.isin(typ_i, self.covariance.test_types) for typ_i in typ[4:]]):
            typ = "opt_kkkk"    # if any test types are detected, it is assumed all observables are kappa
        for iii, L3 in enumerate(Ls):
            L2 = Ls2[None, :]
            L3_vec = vector.obj(rho=L3, phi=0)
            L2_vec = vector.obj(rho=L2, phi=thetas[:, None])
            L1_vec = -L3_vec - L2_vec
            L1 = L1_vec.rho
            w = np.ones(np.shape(L1))
            w[L1 > Lmax] = 0
            w[L1 < Lmin] = 0
            thetas12 = L1_vec.deltaphi(L2_vec)
            bi1 = self.bi.get_bispectrum(typ[4:6] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            bi2 = self.bi.get_bispectrum(typ[6:] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            covs = C1_spline(L1) * C2_spline(L2) / C3_spline(L3)
            I_tmp = dL2 * 2 * np.sum(L2 * w * dTheta * bi1 * bi2 * covs)
            I[iii] = 2 * np.pi * L3 * I_tmp
        I *= 0.5 * f_sky / np.pi * 1 / ((2 * np.pi) ** 2)
        if save_array:
            self.opt_I_cache += I
        I_spline = InterpolatedUnivariateSpline(Ls, I)
        return I_spline.integral(Lmin, Lmax)

    def _get_optimal_bispectrum_Fisher(self, typs, Lmax, dL, Ls, dL2, Ntheta, f_sky, verbose, nu, gal_bins, save_array, only_bins, gal_distro="LSST_gold"):
        typs = np.char.array(typs)
        Lmin = 30     # 1808.07445 and https://cmb-s4.uchicago.edu/wiki/index.php/Survey_Performance_Expectations
        C_inv = self.covariance.get_C_inv(typs, Lmax, nu, gal_bins, gal_distro=gal_distro)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        F = 0
        perms = 0
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                if only_bins and combos[iii] != combos[jjj]:
                    F_tmp = 0
                elif Ls is not None:
                    F_tmp = self._get_optimal_bispectrum_Fisher_sample(typs, typ, Ls, dL2, Ntheta, f_sky, C_inv, nu, gal_bins, save_array, gal_distro=gal_distro)
                else:
                    F_tmp = self._get_optimal_bispectrum_Fisher_element_vec(typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_bins, gal_distro=gal_distro)
                if combos[iii] != combos[jjj]:
                    factor = 2
                else:
                    factor = 1
                perms += factor
                F += factor * F_tmp
                if verbose:
                    print(f"type = {typ}")
                    print(f"F = {F_tmp}")
                    print(f"count = {perms}")
        if perms != np.size(typs)**4:
            raise ValueError(f"{perms} permutations computed, should be {np.size(typs)**4}")
        if save_array:
            self.opt_F = self.opt_I_cache
            self.opt_I_cache = None
        return F

    def _get_F_L_element_sample(self, typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline, gal_distro, Lmin, Lmax):
        thetas, dTheta, weights, C1_spline, C2_spline = self._integral_prep_F_L(Lmax, Ntheta, typ, typs, C_inv)
        F_L = np.zeros(np.size(Ls))
        # Ls2 = self.covariance.get_log_sample_Ls(Lmin, Lmax, Nell2)
        Ls2 = np.arange(Lmin, Lmax+1, dL2)
        if any([np.isin(typ_i, self.covariance.test_types) for typ_i in typ[4:]]):
            typ = "opt_kkkk"    # if any test types are detected, it is assumed all observables are kappa
        if np.size(Ls) == 1:
            Ls = np.array([Ls])
        for iii, L3 in enumerate(Ls):
            I_tmp = np.zeros(np.size(Ls2))
            for jjj, L2 in enumerate(Ls2):
                L3_vec = vector.obj(rho=L3, phi=0)
                L2_vec = vector.obj(rho=L2, phi=thetas)
                L1_vec = L3_vec - L2_vec
                L1 = L1_vec.rho
                w = np.ones(np.shape(L1))
                w[L1 > Lmax] = 0
                w[L1 < Lmin] = 0
                thetas12 = L1_vec.deltaphi(L2_vec)
                bi1 = self.bi.get_bispectrum(typ[4:6] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
                bi2 = self.bi.get_bispectrum(typ[6:] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
                covs = C1_spline(L1) * C2_spline(L2)
                I_tmp[jjj] = 2 * np.sum(L2 * w * dTheta * bi1 * bi2 * covs)
            F_L[iii] = InterpolatedUnivariateSpline(Ls2, I_tmp).integral(Lmin, Lmax) / (2 * C_omega_spline(L3))
        F_L *= 1 / ((2 * np.pi) ** 2)
        return F_L

    def _get_F_L(self, typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro, use_cache, Lmin, Lmax):
        if Lmax is None: Lmax = int(np.ceil(np.max(Ls)))
        if Lmin is None: Lmin = int(np.floor(np.min(Ls)))
        typs = np.char.array(typs)
        if use_cache:
            C_inv = self.C_inv
            C_omega_spline = self.C_omega_spline
        else:
            C_inv = self.covariance.get_C_inv(typs, int(np.ceil(np.max(Ls))), nu, gal_bins, gal_distro=gal_distro)
            omega_ells = self.covariance.get_log_sample_Ls(2, Lmax, 100, dL_small=2)
            C_omega = omega_ps(omega_ells)
            C_omega_spline = InterpolatedUnivariateSpline(omega_ells, C_omega)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        F_L = np.zeros(np.size(Ls))
        perms = 0
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                F_L_tmp = self._get_F_L_element_sample(typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline, gal_distro, Lmin, Lmax)
                if combos[iii] != combos[jjj]:
                    factor = 2
                else:
                    factor = 1
                perms += factor
                F_L += factor * F_L_tmp
        if perms != np.size(typs) ** 4:
            raise ValueError(f"{perms} permutations computed, should be {np.size(typs) ** 4}")
        if return_C_inv:
            return Ls, F_L, C_inv
        return Ls, F_L

    def get_bispectrum_Fisher(self, typ, Lmax=4000, dL=2, Ls=None, dL2=2, Ntheta=10, f_sky=1, arr=False, Lmin=30, nu=353e9, gal_bins=(None,None,None,None), include_N0_kappa="both", gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        typ
        Lmax
        dL
        Ls
        dL2
        Ntheta
        f_sky
        arr
        Lmin
        nu
        gal_bins
        include_N0_kappa

        Returns
        -------

        """
        self.bi.check_type(typ)
        if Ls is not None:
            return self._get_bispectrum_Fisher_sample(typ, Ls, dL2, Ntheta, f_sky, arr, nu=nu, gal_bins=gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        return self._get_bispectrum_Fisher_vec(typ, Lmax, dL, Ntheta, f_sky, Lmin=Lmin, nu=nu, gal_bins=gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)

    def get_F_L(self, typs, Ls, dL2=2, Ntheta=1000, nu=353e9, gal_bins=(None,None,None,None), return_C_inv=False, gal_distro="LSST_gold", use_cache=False, Lmin=None, Lmax=None):
        """

        Parameters
        ----------
        typs
        Ntheta
        nu
        gal_bins

        Returns
        -------

        """
        typs = list(typs)
        return self._get_F_L(typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro, use_cache, Lmin, Lmax)

    def get_optimal_bispectrum_Fisher(self, typs="kg", Lmax=4000, dL=2, Ls=None, dL2=2, Ntheta=10, f_sky=1, verbose=False, nu=353e9, gal_bins=(None,None,None,None), save_array=False, only_bins=False, gal_distro="LSST_gold"):
        """

        Parameters
        ----------
        Lmax
        dL
        Ntheta
        f_sky

        Returns
        -------

        """
        typs = list(typs)
        return self._get_optimal_bispectrum_Fisher(typs, Lmax, dL, Ls, dL2, Ntheta, f_sky, verbose, nu, gal_bins, save_array, only_bins, gal_distro=gal_distro)

    def get_rotation_ps_Fisher(self, Lmax, M_path, f_sky=1, auto=True, camb=False, cmb=True, Lmin=30):
        """
        TODO: Check equation !!!!!!!!!!!
        Parameters
        ----------
        Lmax
        f_sky
        auto

        Returns
        -------

        """
        if camb:
            ells, Cl = self.power.get_camb_postborn_omega_ps(Lmax)
        else:
            # ells = np.arange(2, Lmax + 3, 50)
            ells = np.concatenate((np.arange(2,40,10), np.logspace(1, 3, 50)*4))
            Cl = omega_ps(ells, M_path, cmb=cmb)
        Cl_spline = InterpolatedUnivariateSpline(ells, Cl)
        ells = np.arange(Lmin, Lmax + 1)
        if cmb:
            N0 = self.covariance.noise.get_N0("omega", Lmax)
        else:
            N0 = self.covariance.noise.get_shape_N()
        var = self.power.get_ps_variance(ells, Cl_spline(ells), N0[ells], auto)
        return f_sky * np.sum(Cl_spline(ells) ** 2 / var)

    def reset_noise(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        self.covariance.setup_cmb_noise(self.exp, self.qe, self.gmv, self.ps, self.L_cuts[0], self.L_cuts[1], self.L_cuts[2], self.L_cuts[3], self.iter, self.data_dir)
