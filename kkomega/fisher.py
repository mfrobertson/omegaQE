import numpy as np
from bispectra import Bispectra
from powerspectra import Powerspectra
from noise import Noise
from maths import Maths
from modecoupling import Modecoupling
from scipy.interpolate import InterpolatedUnivariateSpline
from sympy.matrices import Matrix
from sympy import lambdify
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
    noise : Noise
        Instance of Noise, instantiated with the supplied  N0_file and offset.
    N0_ell_factors : bool
        Whether the noise is required to be multiplied by (1/4)(ell + 1/2)^4 during calculations. Note this should be False if the noise supplied is already in the form of N_kappa and N_omega.
    bi : Bispetrum
        Instance of Bispectrum, this instantiation will spline the supplied modecoupling matrix.
    power : Powerspectra
    """

    def __init__(self, N0_file, N0_offset=0, N0_ell_factors=True):
        """
        Constructor

        Parameters
        ----------
        N0_file : str
            Path to .npy file containing the convergence noise in row 0, and the curl noise in row 1. (The same format as Lensit)
        N0_offset : int
            Essentially the value of ellmin in the N0_file. If the first column represents ell = 2, set offset to 2.
        N0_ell_factors : bool
            Whether to multiply the noise by (1/4)(ell + 1/2)^4
        """
        self._setup_noise(N0_file, N0_offset, N0_ell_factors)
        self.bi = Bispectra()
        self.power = Powerspectra()
        self._maths = Maths()
        self.opt_I_cache = None
        self.binned_gal_types = list("abcdef")
        self.test_types = list("xyz")

    def _setup_noise(self, N0_file, N0_offset, N0_ell_factors):
        self.noise = Noise()
        self.noise.setup_cmb_noise(N0_file, N0_offset)
        self.N0_ell_factors = N0_ell_factors

    def _files_match(self, ell_file, M_file):
        if ell_file[:-8] == M_file[:-5]:
            return True
        return False

    def _check_path(self, path):
        if not path_exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

    def setup_bispectra(self, path, ellmax=4000, Nell=100):    #4100 200
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
            N0_omega = self.noise.get_N0("curl", ellmax, ell_factors=self.N0_ell_factors)
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
            return self._get_Cl(typ, ellmax, nu, gal_bins, use_bins, gal_distro=gal_distro)
        if typ[0] == "k":
            N = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
        elif typ[0] in self.test_types:
            N = 3*self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
        elif typ[0] == "I":
            N_cib = self.noise.get_cib_shot_N(ellmax=ellmax, nu=nu)
            N_dust = self.noise.get_dust_N(ellmax=ellmax, nu=nu)
            N = N_cib + N_dust
            N[:111] = 1e10
            N[2001:] = 1e10
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

    def _get_Covs(self, typ, Lmax, all_splines=False, nu=353e9, gal_bins=(None,None,None,None), include_N0_kappa="both", gal_distro="LSST_gold"):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
        C3_spline = N0_omega_spline
        if typ == "kkw":
            if include_N0_kappa == "both":
                C1 = self._get_Cov("kk", Lmax)
                C2 = copy.deepcopy(C1)
            elif include_N0_kappa == "one":
                N0_kappa = self.noise.get_N0("phi", Lmax, tidy=True, ell_factors=self.N0_ell_factors)
                Cl_kappa = self._get_Cl_kappa(Lmax)
                C1 = Cl_kappa + (0.5*N0_kappa)
                C2 = Cl_kappa
            else:
                Cl_kappa = self._get_Cl_kappa(Lmax)
                C1 = Cl_kappa
                C2 = Cl_kappa
        else:
            typ1 = typ[0]
            typ2 = typ[1]
            C1 = self._get_Cov(typ1 + typ1, Lmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            C2 = self._get_Cov(typ2 + typ2, Lmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

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
        print(C_sym)
        if Ntyps > 3:
            C_inv = C_sym.inv('LU')
        else:
            C_inv = C_sym.inv()
        C_inv_func = lambdify(args_no_fI, C_inv)
        Covs = [self._get_Cov(arg, Lmax, nu, gal_bins, gal_distro=gal_distro) for arg in args]
        return C_inv_func(*Covs)

    def _get_optimal_Ns_sympy(self, Lmax, typ, typs, C_inv, all_spline=False):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, tidy=True, ell_factors=self.N0_ell_factors))
        cov3_spline = N0_omega_spline

        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        if all_spline:
            cov_inv1_spline = self._interpolate(cov_inv1)
            cov_inv2_spline = self._interpolate(cov_inv2)
            return cov_inv1_spline, cov_inv2_spline, cov3_spline
        return cov_inv1, cov_inv2, cov3_spline

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
                Cl = self._get_Cl(typ[:2], Lmax, nu, gal_bins, gal_distro=gal_distro)
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

    def _get_bispectrum_Fisher_sample(self, typ, Ls, dL2, Ntheta, f_sky, arr, include_N0_kappa, nu, gal_bins, gal_distro="LSST_gold"):
        Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta,typ, nu, gal_bins, include_N0_kappa=include_N0_kappa, gal_distro=gal_distro)
        Cl_xy_spline = self._interpolate(self._get_Cl(typ[:2], Lmax, nu, gal_bins, gal_distro=gal_distro))
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
        C_inv = self._get_C_inv(typs, Lmax, nu, gal_bins, gal_distro=gal_distro)
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

    def _get_F_L_element_sample(self, typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline, gal_distro="LSST_gold"):
        Lmax, Lmin, _, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta, typ, nu, gal_bins, typs, C_inv, gal_distro=gal_distro)
        F_L = np.zeros(np.size(Ls))
        Ls2 = np.arange(Lmin, Lmax + 1, dL2)
        for iii, L3 in enumerate(Ls):
            L2 = Ls2[None, :]
            L3_vec = vector.obj(rho=L3, phi=0)
            L2_vec = vector.obj(rho=L2, phi=thetas[:,None])
            L1_vec = L3_vec + L2_vec
            L1 = L1_vec.rho
            w = np.ones(np.shape(L1))
            w[L1 > Lmax] = 0
            w[L1 < Lmin] = 0
            thetas12 = L1_vec.deltaphi(L2_vec)
            bi1 = self.bi.get_bispectrum(typ[4:6] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            bi2 = self.bi.get_bispectrum(typ[6:] + "w", L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
            covs = C1_spline(L1) * C2_spline(L2)
            I_tmp = dL2 * 2 * np.sum(L2 * w * dTheta * bi1 * bi2 * covs)
            F_L[iii] = I_tmp / (2 * C_omega_spline(L3))
        F_L *= 1 / ((2 * np.pi) ** 2)
        return F_L


    def _get_F_L(self, typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro="LSST_gold"):
        typs = np.char.array(typs)
        Lmax = np.int(np.max(Ls))
        C_inv = self._get_C_inv(typs, Lmax, nu, gal_bins, gal_distro=gal_distro)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        F_L = np.zeros(np.size(Ls))
        perms = 0
        omega_ells, C_omega = self.power.get_camb_postborn_omega_ps(Lmax * 2)
        C_omega_spline = InterpolatedUnivariateSpline(omega_ells, C_omega)
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                F_L_tmp = self._get_F_L_element_sample(typs, typ, Ls, dL2, Ntheta, C_inv, nu, gal_bins, C_omega_spline, gal_distro=gal_distro)
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

    def get_F_L(self, typs, Ls, dL2=2, Ntheta=100, nu=353e9, gal_bins=(None,None,None,None), return_C_inv=False, gal_distro="LSST_gold"):
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
        return self._get_F_L(typs, Ls, dL2, Ntheta, nu, gal_bins, return_C_inv, gal_distro=gal_distro)

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

    def get_C_inv(self, typs, Lmax, nu, gal_bins=(None,None,None,None)):
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
        return self._get_C_inv(np.char.array(list(typs)), Lmax, nu, gal_bins)

    def get_Cov(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold"):
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

    def get_Cl(self, typ, ellmax, nu=353e9, gal_bins=(None,None,None,None), use_bins=False, gal_distro="LSST_gold"):
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

    def _get_postborn_omega_ps(self, Ls, M_path, Nell_prim, Ntheta, M_ellmax, M_Nell, cmb):
        mode = Modecoupling()
        sep = getFileSep()
        mode_typ = "ww" if cmb else "rr"
        ells_sample = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "ells.npy")
        M = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "M.npy")
        M_spline = mode.spline(ells_sample, M)
        dTheta = np.pi / Ntheta
        thetas = np.linspace(dTheta, np.pi, Ntheta, dtype=float)
        samp1_1 = np.arange(3, 80, 1)
        samp2_1 = np.logspace(1, 3, Nell_prim-77) * 8
        Lprims = np.concatenate((samp1_1, samp2_1))
        I = np.zeros(np.size(Ls))
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            I_tmp = np.zeros(np.size(Lprims))
            for jjj, Lprim in enumerate(Lprims):
                Lprim_vec = vector.obj(rho=Lprim, phi=thetas)
                Lprimprim_vec = L_vec - Lprim_vec
                Lprimprims = Lprimprim_vec.rho
                I_tmp[jjj] = np.sum(2 * Lprim * dTheta * (L*Lprim*np.sin(thetas))** 2 * (Lprim_vec@Lprimprim_vec)** 2 / ((Lprim + 0.5) ** 4 * (Lprimprims + 0.5) ** 4) * M_spline.ev(Lprim, Lprimprims))
            I[iii] = InterpolatedUnivariateSpline(Lprims, I_tmp).integral(3,8000)
        return 4 * I / ((2 * np.pi) ** 2)

    def get_postborn_omega_ps(self, ells, M_path=None, Nell_prim=2000, Ntheta=1000, cmb=True):
        """

        Parameters
        ----------
        ells
        Nell_prim
        Ntheta

        Returns
        -------

        """
        return self._get_postborn_omega_ps(ells, M_path, Nell_prim, Ntheta, 8000, 100, cmb)

    def get_rotation_ps_Fisher(self, Lmax, M_path, f_sky=1, auto=True, camb=False, cmb=True):
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
            ells = np.arange(2, Lmax + 3, 50)
            Cl = self.get_postborn_omega_ps(ells, M_path, cmb=cmb)
        Cl_spline = InterpolatedUnivariateSpline(ells, Cl)
        ells = np.arange(2, Lmax + 1)
        if cmb:
            N0 = self.noise.get_N0("curl", Lmax, True, self.N0_ell_factors)
        else:
            N0 = self.noise.get_shape_N()
        var = self.power.get_ps_variance(ells, Cl_spline(ells), N0[ells], auto)
        return f_sky * np.sum(Cl_spline(ells) ** 2 / var)

    def reset_noise(self, N0_file, N0_offset=0, N0_ell_factors=True):
        """

        Parameters
        ----------
        N0_file
        N0_offset
        N0_ell_factors

        Returns
        -------

        """
        self._setup_noise(N0_file, N0_offset, N0_ell_factors)
