import numpy as np
import omegaqe
from omegaqe.bispectra import Bispectra
from omegaqe.covariance import Covariance
import omegaqe.postborn as pb
from scipy.interpolate import InterpolatedUnivariateSpline
from omegaqe.tools import getFileSep, path_exists
from copy import deepcopy
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

    def __init__(self, exp="SO", qe="TEB", gmv=True, ps="gradient", L_cuts=(30, 3000, 30, 5000), iter=False,
                 iter_ext=False, data_dir=omegaqe.DATA_DIR, setup_bispectra=False, cosmology=None):
        """
        Constructor

        Parameters
        ----------

        """
        self.covariance = Covariance(cosmology=cosmology)
        self.setup_noise(exp, qe, gmv, ps, L_cuts, iter, iter_ext, data_dir)
        self.bi = Bispectra(powerspectra=self.covariance.power)
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

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, iter_ext=None, data_dir=None):
        if exp is not None: self.exp = exp
        if qe is not None: self.qe = qe
        if gmv is not None: self.gmv = gmv
        if ps is not None: self.ps = ps
        if L_cuts is not None: self.L_cuts = L_cuts
        if iter is not None: self.iter = iter
        if iter_ext is not None: self.iter_ext = iter_ext
        if data_dir is not None: self.data_dir = data_dir
        self.reset_noise()

    def setup_bispectra(self, path=f"{omegaqe.CACHE_DIR}{getFileSep()}_M", ellmax=5000, Nell=200):
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
            dir = path + sep + M_type
            self._check_path(dir)
            dir += sep + M_dir + sep
            self._check_path(dir)
            ells = np.load(f"{dir}ells.npy")
            M = np.load(f"{dir}M.npy")
            self.bi.build_M_spline(M_type, ells, M)

    def _get_third_L(self, L1, L2, theta):
        # Using cosine rule (remember that theta is not same as internal angle of bispectrum traingle)
        return np.sqrt(L1 ** 2 + L2 ** 2 + (2 * L1 * L2 * np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample[1:], arr[1:])

    def _get_Covs(self, typ, Lmax, all_splines=False, nu=353e9, gal_bins=(None, None, None, None),
                  include_N0_kappa="both", gal_distro="LSST_gold"):
        N0_omega_spline = self._interpolate(self.covariance.noise.get_N0("omega", Lmax))
        C3_spline = N0_omega_spline
        if typ == "kkw":
            if include_N0_kappa == "both":
                C1 = self.covariance.get_Cov("kk", Lmax)
                C2 = copy.deepcopy(C1)
            elif include_N0_kappa == "one":
                N0_kappa = self.covariance.noise.get_N0("kappa", Lmax)
                Cl_kappa = self.covariance.get_Cl("kk", Lmax)
                C1 = Cl_kappa + (0.5 * N0_kappa)
                C2 = Cl_kappa
            else:
                Cl_kappa = self.covariance.get_Cl("kk", Lmax)
                C1 = Cl_kappa
                C2 = Cl_kappa
        elif typ == "kkk":
            C1 = self.covariance.get_Cov("kk", Lmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            C2 = C1
            C3_spline = self._interpolate(C1)
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

    def change_cosmology(self, param=None, dx=None, minus=False, dx_absolute=False, H0=False):
        default_dx = 0.01
        dx = default_dx if dx is None else dx
        if minus: dx *= -1
        cosmo = self.bi._mode._powerspectra.cosmo
        pars_dict = cosmo.get_pars_dict(cosmo.get_params())
        dx_fac = 1
        dx_log = False
        if param is not None:
            if param == "sig8":
                param = "As"
                dx_fac = pars_dict["sig8"] / (2*pars_dict["As"])
            elif param[:2] == "ln":
                param = param[2:]
                dx_log = True
            elif param[:3] == "100":
                param = param[3:]
                dx_fac = 100
            default_param = pars_dict[param]
            if not dx_absolute: dx *= default_param
            if dx == 0: dx = default_dx
            pars_dict[param] += dx
        cosmo.modify_params(cosmo._pars, pars_dict, H0)
        matter_PK = cosmo.get_matter_PK(typ="matter")
        self.bi._mode.matter_PK = matter_PK
        self.bi._mode._powerspectra.matter_PK = matter_PK
        self.bi.init_M_splines()
        self.covariance.power = self.bi._mode._powerspectra
        self.power = self.covariance.power
        if dx_log:
            return np.log((default_param + dx)/default_param)
        return dx * dx_fac


    def _get_bispectrum(self, typ, L1, L2, L3=None, theta=None, param_dx=(None, None), zmin=0, zmax=None, nu=353e9,
                        gal_bins=(None, None, None, None), gal_distro="LSST_gold", H0=False, lens_delta=False, include_lss=False, is_lss=False):
        param, dx = param_dx
        if param is None:
            if is_lss:
                return self.bi.get_lss_bispectrum(typ, L1, L2, L3, theta, zmin, zmax, nu, gal_bins, gal_distro)
            return self.bi.get_bispectrum(typ, L1, L2, L3, theta, True, zmin, zmax, nu, gal_bins, gal_distro, lens_delta=lens_delta, include_lss=include_lss)
        if is_lss:
            raise ValueError("Doing cosmology fisher with LSS bispectrum? Not sure how you got here....")
        self.change_cosmology(param, dx, True, H0=H0)
        bi_x_h_minus = self.bi.get_bispectrum(typ, L1, L2, L3, theta, True, zmin, zmax, nu, gal_bins, gal_distro, lens_delta=lens_delta, include_lss=include_lss)
        h = self.change_cosmology(param, dx, H0=H0)
        bi_x_h = self.bi.get_bispectrum(typ, L1, L2, L3, theta, True, zmin, zmax, nu, gal_bins, gal_distro, lens_delta=lens_delta, include_lss=include_lss)
        self.change_cosmology(H0=H0)
        return (bi_x_h - bi_x_h_minus) / (2 * np.abs(h))

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _get_Cov(self, typ, Lmax):
        if typ == "ww":
            return self.covariance.noise.get_N0("omega", Lmax)
        elif "w" in typ:
            return np.zeros(Lmax + 1)
        return self.covariance.get_Cov(typ, Lmax)

    def _get_denom_parts(self, typ, Lmax):
        Cov0 = self._get_Cov(typ[0] + typ[0], Lmax)
        Cov1 = self._get_Cov(typ[1] + typ[1], Lmax)
        Cov2 = self._get_Cov(typ[2] + typ[2], Lmax)
        Cl01 = self._get_Cov(typ[0] + typ[1], Lmax)
        Cl02 = self._get_Cov(typ[0] + typ[2], Lmax)
        Cl12 = self._get_Cov(typ[1] + typ[2], Lmax)

        Cov0_spline = self._interpolate(Cov0)
        Cov1_spline = self._interpolate(Cov1)
        Cov2_spline = self._interpolate(Cov2)
        Cl01_spline = self._interpolate(Cl01)
        Cl02_spline = self._interpolate(Cl02)
        Cl12_spline = self._interpolate(Cl12)
        return Cov0, Cov1, Cov2, Cl01, Cl12, Cl02, Cov0_spline, Cov1_spline, Cov2_spline, Cl01_spline, Cl02_spline, Cl12_spline

    def _get_denom_vec(self, Cov0, Cov1, Cov2, Cl01, Cl12, Cl02, Cov0_spline, Cov1_spline, Cov2_spline, Cl01_spline, Cl02_spline, Cl12_spline, Ls, L3):
        denom = Cov0[None, Ls, None] * Cov1[None, None, Ls] * Cov2_spline(L3)
        denom += Cov0[None, Ls, None] * Cl12[None, None, Ls] * Cl12_spline(L3)
        denom += Cl01[None, Ls, None] * Cl12[None, None, Ls] * Cl02_spline(L3)
        denom += Cl01[None, Ls, None] * Cl01[None, None, Ls] * Cov2_spline(L3)
        denom += Cl02[None, Ls, None] * Cl01[None, None, Ls] * Cl12_spline(L3)
        denom += Cl02[None, Ls, None] * Cov1[None, None, Ls] * Cl02_spline(L3)
        return denom

    def _get_denom_samp(self, Cov0, Cov1, Cov2, Cl01, Cl12, Cl02, Cov0_spline, Cov1_spline, Cov2_spline, Cl01_spline, Cl02_spline, Cl12_spline, L1, L2, L3):
        denom = Cov0_spline(L1) * Cov1_spline(L2) * Cov2_spline(L3)
        denom += Cov0_spline(L1) * Cl12_spline(L2) * Cl12_spline(L3)
        denom += Cl01_spline(L1) * Cl12_spline(L2) * Cl02_spline(L3)
        denom += Cl01_spline(L1) * Cl01_spline(L2) * Cov2_spline(L3)
        denom += Cl02_spline(L1) * Cl01_spline(L2) * Cl12_spline(L3)
        denom += Cl02_spline(L1) * Cov1_spline(L2) * Cl02_spline(L3)
        return denom

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ, Lmin, nu, gal_bins, typs=None, C_inv=None, include_N0_kappa="both", gal_distro="LSST_gold"):
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(Lmin, Lmax + 1, dL)
        L3 = self._get_third_L(Ls[:, None], Ls[None, :], thetas[:, None, None])
        w = np.ones(np.shape(L3))
        w[L3 < Lmin] = 0
        w[L3 > Lmax] = 0
        if typ[:3] == "opt":
            C1, C2, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv)
            denom = C1[None, Ls, None] * C2[None, None, Ls] / C3_spline(L3)  # These are actually the C_inv
        else:
            denom_parts = self._get_denom_parts(typ, Lmax)
            denom = self._get_denom_vec(*denom_parts, Ls, L3)
        return Ls, L3, dTheta, w, denom

    def _get_bispectrum_Fisher_vec(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa, Lmin, nu, gal_bins,
                                   gal_distro="LSST_gold", param=None, dx=None, lens_delta=False, include_lss=False):
        # TODO: Only works for rotation bispectra! (Don't want L3 as input to M-spines)
        Ls, L3, dTheta, w, denom = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin, nu, gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        if np.size(param) == 2:
            param1 = param[0]
            param2 = param[1]
            bi1 = self._get_bispectrum(typ, Ls[:, None], Ls[None, :], L3, param_dx=(param1, dx), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro, lens_delta=lens_delta, include_lss=include_lss)
            bi2 = self._get_bispectrum(typ, Ls[:, None], Ls[None, :], L3, param_dx=(param2, dx), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro, lens_delta=lens_delta, include_lss=include_lss)
        else:
            bi1 = bi2 = self._get_bispectrum(typ, Ls[:, None], Ls[None, :], L3, param_dx=(param, dx), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro, lens_delta=lens_delta, include_lss=include_lss)
        I = 2 * 2 * np.pi * dL * dL * np.sum(
            Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bi1 * bi2 / denom)
        return f_sky / np.pi * I / ((2 * np.pi) ** 2)

    def _integral_prep_sample(self, Ls, Ntheta, typ, nu, gal_bins, typs=None, C_inv=None, include_N0_kappa="both", gal_distro="LSST_gold"):
        Lmax = int(np.max(Ls))
        Lmin = int(np.min(Ls))
        if typ[:3] == "opt":
            C1_spline, C2_spline, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv, all_spline=True)
        else:
            C1_spline, C2_spline, C3_spline = self._get_Covs(typ, Lmax, all_splines=True, nu=nu, gal_bins=gal_bins,include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[:-1] = Ls[1:] - Ls[0:-1]
        dLs[-1] = dLs[-2]
        return Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    def _integral_prep_F_L(self, Lmax, Ntheta, typ, typs, C_inv):
        C1_spline, C2_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv, all_spline=True,
                                                          return_cov3=False)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        return thetas, dTheta, weights, C1_spline, C2_spline

    def _get_bispectrum_Fisher_sample(self, typ, Ls, dL2, Ntheta, f_sky, arr, include_N0_kappa, nu, gal_bins, gal_distro="LSST_gold", param=None, dx=None, lens_delta=False, include_lss=False, is_lss=False):
        Lmax, Lmin, dLs, thetas, dTheta, weights, _, _, _ = self._integral_prep_sample(Ls,Ntheta,typ, nu,gal_bins,include_N0_kappa=include_N0_kappa,gal_distro=gal_distro)
        denom_parts = self._get_denom_parts(typ, Lmax)
        I = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax + 1, dL2)
        L2 = Ls2[None, :]
        L2_vec = vector.obj(rho=L2, phi=thetas[:, None])
        for iii, L3 in enumerate(Ls):
            I_tmp = 0
            L3_vec = vector.obj(rho=L3, phi=0)
            L1_vec = L3_vec - L2_vec if typ[:-1] == "w" else -L3_vec - L2_vec   # Not sure this matters much
            L1 = L1_vec.rho
            w = np.ones(np.shape(L1))
            w[L1 > Lmax] = 0
            w[L1 < Lmin] = 0
            bispectrum = self._get_bispectrum(typ, L1, L2, L3, param_dx=(param, dx), nu=nu, gal_bins=gal_bins, gal_distro=gal_distro, lens_delta=lens_delta, include_lss=include_lss, is_lss=is_lss)
            # thetas12 = L1_vec.deltaphi(L2_vec)
            # bispectrum = self._get_bispectrum(typ, L1, L2, theta=thetas12, param_dx=(param, dx), nu=nu, gal_bins=gal_bins, gal_distro=gal_distro, lens_delta=lens_delta, include_lss=include_lss, is_lss=is_lss)
            denom = self._get_denom_samp(*denom_parts, L1, L2, L3)
            I_tmp += dL2 * 2 * np.sum(L2 * w * dTheta * bispectrum ** 2 / denom)
            I[iii] = 2 * np.pi * L3 * I_tmp
        I *= f_sky / np.pi * 1 / ((2 * np.pi) ** 2)
        if arr:
            return I
        I_spline = InterpolatedUnivariateSpline(Ls, I)
        return I_spline.integral(Lmin, Lmax)

    def _get_optimal_bispectrum_Fisher_element_vec(self, typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_bins,
                                                   gal_distro="LSST_gold", param=None, dx=None, H0=False):
        Ls, L3, dTheta, w, covs = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin=Lmin, nu=nu, gal_bins=gal_bins,
                                                          typs=typs, C_inv=C_inv, gal_distro=gal_distro)
        if np.size(param) == 2:
            param1 = param[0]
            param2 = param[1]
        else:
            param1 = param2 = param
        bi1 = self._get_bispectrum(typ[4:6] + "w", Ls[:, None], Ls[None, :], L3, param_dx=(param1, dx), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro, H0=H0)
        bi2 = self._get_bispectrum(typ[6:] + "w", Ls[:, None], Ls[None, :], L3, param_dx=(param2, dx), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro, H0=H0)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bi1 * bi2 * covs)
        return 0.5 * f_sky / np.pi * I / ((2 * np.pi) ** 2)

    def _get_optimal_bispectrum_Fisher_sample(self, typs, typ, Ls, dL2, Ntheta, f_sky, C_inv, nu, gal_bins, save_array, gal_distro="LSST_gold", param=None, dx=None):
        Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls,Ntheta,typ, nu,gal_bins,typs=typs,C_inv=C_inv,gal_distro=gal_distro)
        if save_array and self.opt_I_cache is None:
            self.opt_I_cache = np.zeros(np.size(Ls))
            self.opt_Ls = Ls
        I = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax + 1, dL2)
        if any([np.isin(typ_i, self.covariance.test_types) for typ_i in typ[4:]]):
            typ = "opt_kkkk"  # if any test types are detected, it is assumed all observables are kappa
        if np.size(param) == 2:
            param1 = param[0]
            param2 = param[1]
            if dx is not None and np.size(dx) == 2:
                dx1 = dx[1]
                dx2 = dx[2]
            else:
                dx1 = dx2 = dx
        else:
            param1 = param2 = param
            dx1 = dx2 = dx
        for iii, L3 in enumerate(Ls):
            L2 = Ls2[None, :]
            L3_vec = vector.obj(rho=L3, phi=0)
            L2_vec = vector.obj(rho=L2, phi=thetas[:, None])
            L1_vec = L3_vec - L2_vec
            L1 = L1_vec.rho
            w = np.ones(np.shape(L1))
            w[L1 > Lmax] = 0
            w[L1 < Lmin] = 0
            thetas12 = L1_vec.deltaphi(L2_vec)
            bi1 = self._get_bispectrum(typ[4:6] + "w", L1, L2, theta=thetas12, param_dx=(param1, dx1), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
            bi2 = self._get_bispectrum(typ[6:] + "w", L1, L2, theta=thetas12, param_dx=(param2, dx2), nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
            covs = C1_spline(L1) * C2_spline(L2) / C3_spline(L3)
            I_tmp = dL2 * 2 * np.sum(L2 * w * dTheta * bi1 * bi2 * covs)
            I[iii] = 2 * np.pi * L3 * I_tmp
        I *= 0.5 * f_sky / np.pi * 1 / ((2 * np.pi) ** 2)
        if save_array:
            self.opt_I_cache += I
        I_spline = InterpolatedUnivariateSpline(Ls, I)
        return I_spline.integral(Lmin, Lmax)

    def _get_optimal_bispectrum_Fisher(self, typs, Lmax, dL, Ls, dL2, Ntheta, f_sky, verbose, nu, gal_bins, save_array, only_bins, gal_distro="LSST_gold", param=None, dx=None, H0=False):
        typs = np.char.array(typs)
        Lmin = 30  # 1808.07445 and https://cmb-s4.uchicago.edu/wiki/index.php/Survey_Performance_Expectations
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
                    F_tmp = self._get_optimal_bispectrum_Fisher_sample(typs, typ, Ls, dL2, Ntheta, f_sky, C_inv, nu,gal_bins, save_array, gal_distro=gal_distro,param=param, dx=dx)
                else:
                    F_tmp = self._get_optimal_bispectrum_Fisher_element_vec(typs, typ, Lmax, dL, Ntheta, f_sky, C_inv,Lmin, nu, gal_bins, gal_distro=gal_distro,param=param, dx=dx, H0=H0)
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
        if perms != np.size(typs) ** 4:
            raise ValueError(f"{perms} permutations computed, should be {np.size(typs) ** 4}")
        if save_array:
            self.opt_F = self.opt_I_cache
            self.opt_I_cache = None
        return F
    
    def additional_mu_bispectra(self, typ, L1, L2, L3, M_spline, nu, gal_bins, gal_distro):
        sec_var = typ[-1]
        bi_typ = typ[:-1]
        if "g" not in bi_typ:
            return 0
        if bi_typ == "gg":
            bi = self.bi.get_bispectrum("ug" + sec_var, L1, L2, L3, M_spline=M_spline, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
            bi += self.bi.get_bispectrum("gu" + sec_var, L1, L2, L3, M_spline=M_spline, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
            bi += self.bi.get_bispectrum("uu" + sec_var, L1, L2, L3, M_spline=M_spline, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
            return bi
        return self.bi.get_bispectrum(bi_typ.replace("g", "u") + sec_var, L1, L2, L3, M_spline=M_spline, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)


    def _get_F_L_element_sample(self, typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline, gal_distro, Lmin, Lmax, mag_bias):
        thetas, dTheta, weights, C1_spline, C2_spline = self._integral_prep_F_L(Lmax, Ntheta, typ, typs, C_inv)
        F_L = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax + 1, dL2)
        if any([np.isin(typ_i, self.covariance.test_types) for typ_i in typ[4:]]):
            typ = "opt_kkkk"  # if any test types are detected, it is assumed all observables are kappa
        bi_typ1 = typ[4:6] + "w"
        bi_typ2 = typ[6:] + "w"
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
                # thetas12 = L1_vec.deltaphi(L2_vec)
                # bi1 = self.bi.get_bispectrum(typ[4:6] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
                # bi2 = self.bi.get_bispectrum(typ[6:] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
                bi1 = self.bi.get_bispectrum(bi_typ1, L1, L2, L3, M_spline=True, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
                bi2 = self.bi.get_bispectrum(bi_typ2, L1, L2, L3, M_spline=True, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
                if mag_bias:
                    bi1 += self.additional_mu_bispectra(bi_typ1, L1, L2, L3, M_spline=True, nu=nu,gal_bins=gal_bins, gal_distro=gal_distro)
                covs = C1_spline(L1) * C2_spline(L2)
                I_tmp[jjj] = 2 * np.sum(L2 * w * dTheta * bi1 * bi2 * covs)
            F_L[iii] = InterpolatedUnivariateSpline(Ls2, I_tmp).integral(Lmin, Lmax) / (2 * C_omega_spline(L3))
        F_L *= 1 / ((2 * np.pi) ** 2)
        return F_L

    def _get_F_L(self, typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro, use_cache, Lmin, Lmax, mag_bias):
        if Lmax is None: Lmax = int(np.ceil(np.max(Ls)))
        if Lmin is None: Lmin = int(np.floor(np.min(Ls)))
        typs = np.char.array(typs)
        if mag_bias: self.covariance.mag_bias = True
        if use_cache:
            C_inv = self.C_inv
            C_omega_spline = self.C_omega_spline
        else:
            C_inv = self.covariance.get_C_inv(typs, int(np.ceil(np.max(Ls))), nu, gal_bins, gal_distro=gal_distro)
            # omega_ells = self.covariance.get_log_sample_Ls(2, Lmax, 100, dL_small=2)
            omega_ells = np.geomspace(2, Lmax, 100)
            C_omega = pb.omega_ps(omega_ells)
            C_omega_spline = InterpolatedUnivariateSpline(omega_ells, C_omega)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        F_L = np.zeros(np.size(Ls))
        perms = 0
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                F_L_tmp = self._get_F_L_element_sample(typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline,gal_distro, Lmin, Lmax, mag_bias)
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

    def get_lss_bispectrum_Fisher(self, typ, Ls, dL2=2, Ntheta=20, f_sky=1, arr=False,
                              nu=353e9, gal_bins=(None, None, None, None), include_N0_kappa="both",
                              gal_distro="LSST_gold"):
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
        return self._get_bispectrum_Fisher_sample(typ, Ls, dL2, Ntheta, f_sky, arr, nu=nu, gal_bins=gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro, is_lss=True)

    def get_bispectrum_Fisher(self, typ, Lmax=4000, dL=1, Ls=None, dL2=2, Ntheta=20, f_sky=1, arr=False, Lmin=30,
                              nu=353e9, gal_bins=(None, None, None, None), include_N0_kappa="both",
                              gal_distro="LSST_gold", param=None, dx=None, lens_delta=False, include_lss=False):
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
        if param is not None:
            self.change_cosmology()
        if Ls is not None:
            if param is not None:
                raise RuntimeWarning("Are you sure you want to do param Fisher using sample method?")
            return self._get_bispectrum_Fisher_sample(typ, Ls, dL2, Ntheta, f_sky, arr, nu=nu, gal_bins=gal_bins,
                                                      include_N0_kappa=include_N0_kappa, gal_distro=gal_distro,
                                                      param=param, dx=dx, lens_delta=lens_delta, include_lss=include_lss)
        if typ[-1] != "w":
            raise RuntimeWarning(f"Are you sure you want to vectorized Fisher for type {typ}?")
        return self._get_bispectrum_Fisher_vec(typ, Lmax, dL, Ntheta, f_sky, Lmin=Lmin, nu=nu, gal_bins=gal_bins,
                                               include_N0_kappa=include_N0_kappa, gal_distro=gal_distro, param=param,
                                               dx=dx, lens_delta=lens_delta, include_lss=include_lss)

    def get_F_L(self, typs, Ls, dL2=2, Ntheta=1000, nu=353e9, gal_bins=(None, None, None, None), return_C_inv=False,
                gal_distro="LSST_gold", use_cache=False, Lmin=None, Lmax=None, mag_bias=False):
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
        return self._get_F_L(typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro, use_cache, Lmin, Lmax, mag_bias)

    def get_optimal_bispectrum_Fisher(self, typs="kg", Lmax=4000, dL=2, Ls=None, dL2=2, Ntheta=10, f_sky=1,
                                      verbose=False, nu=353e9, gal_bins=(None, None, None, None), save_array=False,
                                      only_bins=False, gal_distro="LSST_gold", param=None, dx=None, H0=False):
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
        # NOTE: if Ntyps>1 and param1!=param2 then Fisher will be wrong
        typs = list(typs)
        if param is not None:
            self.change_cosmology(H0=H0)
        if param is not None and Ls is not None:
            raise RuntimeWarning("Are you sure you want to do param Fisher using sample method?")
        return self._get_optimal_bispectrum_Fisher(typs, Lmax, dL, Ls, dL2, Ntheta, f_sky, verbose, nu, gal_bins,
                                                   save_array, only_bins, gal_distro=gal_distro, param=param, dx=dx, H0=H0)

    def get_cmb_Fisher(self, Lmax, f_sky=1, Lmin=2, param=None, dx=None, H0=False):
        """

        Parameters
        ----------
        Lmax
        fsky
        Lmin
        param
        dx

        Returns
        -------

        """
        #astro-ph/0603019
        def _get_dCl(param, dx):
            mat = np.zeros((Lmax + 1 - Lmin, 2, 2))
            self.change_cosmology(param, dx, True, H0=H0)
            Cl_t_x_h_minus = self.covariance.power.cosmo.get_lens_ps("TT", ellmax=Lmax)[Lmin:Lmax + 1]
            Cl_te_x_h_minus = self.covariance.power.cosmo.get_lens_ps("TE", ellmax=Lmax)[Lmin:Lmax + 1]
            Cl_e_x_h_minus = self.covariance.power.cosmo.get_lens_ps("EE", ellmax=Lmax)[Lmin:Lmax + 1]
            h = self.change_cosmology(param, dx, H0=H0)
            Cl_t_x_h = self.covariance.power.cosmo.get_lens_ps("TT", ellmax=Lmax)[Lmin:Lmax + 1]
            Cl_te_x_h = self.covariance.power.cosmo.get_lens_ps("TE", ellmax=Lmax)[Lmin:Lmax + 1]
            Cl_e_x_h = self.covariance.power.cosmo.get_lens_ps("EE", ellmax=Lmax)[Lmin:Lmax + 1]
            mat[:, 0, 0] = Cl_t_x_h - Cl_t_x_h_minus
            mat[:, 1, 1] = Cl_e_x_h - Cl_e_x_h_minus
            mat[:, 1, 0] = mat[:, 0, 1] = Cl_te_x_h - Cl_te_x_h_minus
            mat /= (2 * np.abs(h))
            return mat

        self.change_cosmology(H0=H0)
        cov_mat = np.zeros((Lmax + 1 - Lmin, 2, 2))
        N_t = self.covariance.noise.get_cmb_gaussian_N("TT", None, None, ellmax=Lmax, exp=self.exp)
        N_e = self.covariance.noise.get_cmb_gaussian_N("EE", None, None, ellmax=Lmax, exp=self.exp)
        cov_mat[:, 0, 0] = self.covariance.power.cosmo.get_lens_ps("TT", ellmax=Lmax)[Lmin:Lmax + 1]
        cov_mat[:, 1, 1] = self.covariance.power.cosmo.get_lens_ps("EE", ellmax=Lmax)[Lmin:Lmax + 1]
        cov_mat[:, 0, 1] = cov_mat[:, 1, 0] = self.covariance.power.cosmo.get_lens_ps("TE", ellmax=Lmax)[Lmin:Lmax + 1]
        cov_mat[:, 0, 0] += N_t[Lmin:Lmax + 1]
        cov_mat[:, 1, 1] += N_e[Lmin:Lmax + 1]
        inv_cov_mat = np.linalg.pinv(cov_mat)
        if param is None:
            Cl_1 = Cl_2 = deepcopy(cov_mat)
        elif np.size(param) == 2:
            if dx is not None and np.size(dx) == 2:
                Cl_1 = _get_dCl(param[0], dx[0])
                Cl_2 = _get_dCl(param[1], dx[1])
            else:
                Cl_1 = _get_dCl(param[0], dx)
                Cl_2 = _get_dCl(param[1], dx)
        else:
            Cl_1 = Cl_2 = _get_dCl(param, dx)
        leg1 = np.matmul(inv_cov_mat, Cl_1)
        leg2 = np.matmul(inv_cov_mat, Cl_2)
        res = np.matmul(leg1, leg2)
        trace = res[:, 0, 0] + res[:, 1, 1]
        ells = np.arange(Lmin, Lmax + 1)
        self.change_cosmology(H0=H0)
        return f_sky * np.sum(trace * (ells + 0.5))
    
    def get_lss_Fisher(self, typs, Lmax, f_sky=1, Lmin=2, nu=353e9, gal_bins=(None, None, None, None), gal_distro="LSST_gold", param=None, dx=None, H0=False):
        """

        Parameters
        ----------
        Lmax
        fsky
        Lmin
        param
        dx

        Returns
        -------

        """
        #astro-ph/0603019
        def _get_dCl(param, dx):
            self.change_cosmology(param, dx, True, H0=H0)
            cov_mat_x_h_minus = self.covariance.get_Cov_mat(typs, Lmax,  nu, gal_bins, gal_distro=gal_distro, noise=False)
            h = self.change_cosmology(param, dx, H0=H0)
            cov_mat_x_h = self.covariance.get_Cov_mat(typs, Lmax,  nu, gal_bins, gal_distro=gal_distro, noise=False)
            return (cov_mat_x_h - cov_mat_x_h_minus) / (2 * np.abs(h))

        self.change_cosmology(H0=H0)
        cov_mat = self.covariance.get_Cov_mat(typs, Lmax,  nu, gal_bins, gal_distro=gal_distro, noise=False)
        inv_cov_mat = self.covariance.get_C_inv(typs, Lmax, nu, gal_bins, gal_distro=gal_distro)
        if param is None:
            Cl_1 = Cl_2 = deepcopy(cov_mat)
        elif np.size(param) == 2:
            if dx is not None and np.size(dx) == 2:
                Cl_1 = _get_dCl(param[0], dx[0])
                Cl_2 = _get_dCl(param[1], dx[1])
            else:
                Cl_1 = _get_dCl(param[0], dx)
                Cl_2 = _get_dCl(param[1], dx)
        else:
            Cl_1 = Cl_2 = _get_dCl(param, dx)
        
        old_shape = np.shape(inv_cov_mat)
        new_shape = (old_shape[-1], old_shape[0], old_shape[0])
        inv_cov_mat = inv_cov_mat.reshape(new_shape)[Lmin:Lmax + 1]
        Cl_1 = Cl_1.reshape(new_shape)[Lmin:Lmax + 1]
        Cl_2 = Cl_2.reshape(new_shape)[Lmin:Lmax + 1]
        leg1 = np.matmul(inv_cov_mat, Cl_1)
        leg2 = np.matmul(inv_cov_mat, Cl_2)
        res = np.matmul(leg1, leg2)
        trace = res[:, 0, 0] + res[:, 1, 1]
        ells = np.arange(Lmin, Lmax + 1)
        self.change_cosmology(H0=H0)
        return f_sky * np.sum(trace * (ells + 0.5))

    def get_kappa_ps_Fisher(self, Lmax, f_sky=1, auto=True, Lmin=30, param=None, dx=None, H0=False):
        """
        Parameters
        ----------
        Lmax
        f_sky
        auto

        Returns
        -------

        """

        def _get_dCl(param, dx):
            self.change_cosmology(param, dx, True, H0=H0)
            Cl_x_h_minus = self.power.get_kappa_ps(ells)
            h = self.change_cosmology(param, dx, H0=H0)
            Cl_x_h = self.power.get_kappa_ps(ells)
            return (Cl_x_h - Cl_x_h_minus) / (2 * np.abs(h))

        self.change_cosmology(H0=H0)
        ells = np.arange(Lmin, Lmax + 1)
        Cl_kk = self.power.get_kappa_ps(ells)
        if param is None:
            Cl_1 = Cl_2 = Cl_kk
        elif np.size(param) == 2:
            if dx is not None and np.size(dx) == 2:
                Cl_1 = _get_dCl(param[0], dx[0])
                Cl_2 = _get_dCl(param[1], dx[1])
            else:
                Cl_1 = _get_dCl(param[0], dx)
                Cl_2 = _get_dCl(param[1], dx)
        else:
            Cl_1 = Cl_2 = _get_dCl(param, dx)
        N0 = self.covariance.noise.get_N0("kappa", Lmax)
        if auto:
            var = 2 / (2 * ells + 1) * (Cl_kk + N0[ells]) ** 2
        else:
            var = 2 / (2 * ells + 1) * (Cl_kk ** 2 + 0.5 * (N0[ells] * Cl_kk))
        self.change_cosmology(H0=H0)
        return f_sky * np.sum(Cl_1 * Cl_2 / var)
    
    def get_omega_ps_Fisher(self, Lmax, f_sky=1, auto=True, Lmin=30, param=None, dx=None, H0=False, F_L_path=f"{omegaqe.RESULTS_DIR}{getFileSep()}F_L_results"):
        """
        Parameters
        ----------
        Lmax
        f_sky
        auto

        Returns
        -------

        """

        def _get_dCl(param, dx):
            self.change_cosmology(param, dx, True, H0=H0)
            ells_w, cl = self.power.cosmo.get_postborn_omega_ps(acc=3)
            # ells_w = np.geomspace(Lmin, Lmax, 20)
            # cl = omega_ps(ells_w, zmin=0, zmax=1100, powerspectra=self.power)
            Cl_x_h_minus = InterpolatedUnivariateSpline(ells_w, cl)(ells)
            h = self.change_cosmology(param, dx, H0=H0)
            ells_w, cl = self.power.cosmo.get_postborn_omega_ps(acc=3)
            # cl = omega_ps(ells_w, zmin=0, zmax=1100, powerspectra=self.power)
            Cl_x_h = InterpolatedUnivariateSpline(ells_w, cl)(ells)
            return (Cl_x_h - Cl_x_h_minus) / (2 * np.abs(h))

        self.change_cosmology(H0=H0)
        ells = np.arange(Lmin, Lmax + 1)
        ells_w, cl = self.power.cosmo.get_postborn_omega_ps(acc=3)
        Cl_ww = InterpolatedUnivariateSpline(ells_w, cl)(ells)
        # ells_w = np.geomspace(Lmin, Lmax, 20)
        # cl = omega_ps(ells_w, zmin=0, zmax=1100, powerspectra=self.power)
        # Cl_ww = InterpolatedUnivariateSpline(ells_w, cl)(ells)

        gmv_str = "gmv" if self.gmv else "single"
        full_F_L_path = f"{F_L_path}/kgI/{self.exp}/{gmv_str}/{self.qe}/{30}_{3000}/1_2000"
        ells_fl = np.load(f"{full_F_L_path}/Ls.npy")
        F_L = np.load(f"{full_F_L_path}/F_L.npy")
        F_L = InterpolatedUnivariateSpline(ells_fl, F_L)(ells)
        
        if param is None:
            Cl_1 = Cl_2 = Cl_ww
        elif np.size(param) == 2:
            if dx is not None and np.size(dx) == 2:
                Cl_1 = _get_dCl(param[0], dx[0])
                Cl_2 = _get_dCl(param[1], dx[1])
            else:
                Cl_1 = _get_dCl(param[0], dx)
                Cl_2 = _get_dCl(param[1], dx)
        else:
            Cl_1 = Cl_2 = _get_dCl(param, dx)
        N0 = self.covariance.noise.get_N0("omega", Lmax)
        if auto:
            var = 2 / (2 * ells + 1) * (Cl_ww + N0[ells]) ** 2
        else:
            # var = 1 / (2 * ells + 1) * (Cl_ww * N0[ells] / F_L)
            var = 1 / (2 * ells + 1) * ((Cl_ww + N0[ells]) * Cl_ww / F_L + (Cl_ww**2))
        self.change_cosmology(H0=H0)
        return f_sky * np.sum(Cl_1 * Cl_2 / var)

    def get_rotation_ps_Fisher(self, Lmax, M_path, f_sky=1, auto=True, camb=False, cmb=True, Lmin=30, n=40):
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
            ells, Cl = self.power.cosmo.get_postborn_omega_ps()
        else:
            ells = self.covariance.get_log_sample_Ls(2, Lmax, 100, dL_small=2)
            Cl = pb.omega_ps(ells, M_path, cmb=cmb)
        Cl_spline = InterpolatedUnivariateSpline(ells, Cl)
        ells = np.arange(Lmin, Lmax + 1)
        if cmb:
            N0 = self.covariance.noise.get_N0("omega", Lmax)
        else:
            N0 = self.covariance.noise.get_shape_N(n=n)
        if auto:
            var = 2 / (2 * ells + 1) * (Cl_spline(ells) + N0[ells]) ** 2
        else:
            var = 2 / (2 * ells + 1) * (Cl_spline(ells) ** 2 + 0.5 * (N0[ells] * Cl_spline(ells)))
        return f_sky * np.sum(Cl_spline(ells) ** 2 / var)

    def get_len_len_kappa_Fisher(self, Lmax, M_path, f_sky=1, cmb=True, Lmin=30, n=40):
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
        ells = np.geomspace(2, Lmax, 100)
        Cl = pb.len_len_kappa_ps(ells, M_path, cmb=cmb)
        Cl_spline = InterpolatedUnivariateSpline(ells, Cl)
        ells = np.arange(Lmin, Lmax + 1)
        cl_ll_kappa = Cl_spline(ells)
        cl_kappa = self.power.get_kappa_ps(ells)
        if cmb:
            N0 = self.covariance.noise.get_N0("kappa", Lmax)
        else:
            N0 = self.covariance.noise.get_shape_N(n=n)
        return f_sky * np.sum((2*ells + 1) * np.abs(cl_ll_kappa) / (cl_kappa + N0[ells]))

    def reset_noise(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        self.covariance.setup_cmb_noise(self.exp, self.qe, self.gmv, self.ps, self.L_cuts[0], self.L_cuts[1],
                                        self.L_cuts[2], self.L_cuts[3], self.iter, self.iter_ext, self.data_dir)
