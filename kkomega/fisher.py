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

    def setup_bispectra(self, path, ellmax=4000, Nell=1000):
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
        M_types = ["kappa-kappa", "kappa-gal", "gal-kappa", "gal-gal"]
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
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample, arr)

    def _get_Cl_kappa(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_kappa_ps(ells)

    def _get_Cl_gal(self,ellmax, gal_win_zmin=None, gal_win_zmax=None):
        ells = np.arange(ellmax + 1)
        return self.power.get_gal_ps(ells, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)

    def _get_Cl_cib(self,ellmax, nu=857e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_ps(ells, nu=nu)

    def _get_Cl_gal_kappa(self,ellmax, gal_win_zmin=None, gal_win_zmax=None):
        ells = np.arange(ellmax + 1)
        return self.power.get_gal_kappa_ps(ells, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)

    def _get_Cl_cib_kappa(self,ellmax, nu=857e9):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_kappa_ps(ells, nu=nu)

    def _get_Cl_cib_gal(self,ellmax, nu=857e9, gal_win_zmin=None, gal_win_zmax=None):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_gal_ps(ells, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)

    def _get_Cl(self, typ, ellmax, nu=857e9, gal_win_zmin=None, gal_win_zmax=None):
        if typ == "kk":
            return self._get_Cl_kappa(ellmax)
        elif typ == "gk" or typ == "kg":
            return self._get_Cl_gal_kappa(ellmax, gal_win_zmin, gal_win_zmax)
        elif typ == "gg":
            return self._get_Cl_gal(ellmax, gal_win_zmin, gal_win_zmax)
        elif typ == "Ik" or typ == "kI":
            return self._get_Cl_cib_kappa(ellmax, nu)
        elif typ == "II":
            return self._get_Cl_cib(ellmax, nu)
        elif typ == "Ig" or typ == "gI":
            return self._get_Cl_cib_gal(ellmax, nu, gal_win_zmin, gal_win_zmax)
        elif typ == "ww":
            N0_omega = self.noise.get_N0("curl", ellmax, ell_factors=self.N0_ell_factors)
            return N0_omega

    def _get_Cov(self, typ, ellmax, nu=857e9, gal_win_zmin=None, gal_win_zmax=None):
        if typ == "kk":
            N0_kappa = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
            return self._get_Cl_kappa(ellmax) + N0_kappa
        elif typ == "gk" or typ == "kg":
            return self._get_Cl_gal_kappa(ellmax, gal_win_zmin, gal_win_zmax)
        elif typ == "gg":
            N0_gal = self.noise.get_gal_shot_N(ellmax=ellmax)
            return self._get_Cl_gal(ellmax, gal_win_zmin, gal_win_zmax) + N0_gal
        elif typ == "Ik" or typ == "kI":
            return self._get_Cl_cib_kappa(ellmax, nu)
        elif typ == "II":
            N0_cib = self.noise.get_cib_shot_N(ellmax=ellmax, nu=nu)
            N_dust = self.noise.get_dust_N(ellmax=ellmax, nu=nu)
            #N_cmb = self.noise.get_cmb_N(ellmax=ellmax, nu=nu)
            return self._get_Cl_cib(ellmax, nu) + N0_cib + N_dust
        elif typ == "Ig" or typ == "gI":
            return self._get_Cl_cib_gal(ellmax, nu, gal_win_zmin, gal_win_zmax)
        if typ == "xx":
            N0_kappa = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
            return self._get_Cl_kappa(ellmax) + 3*N0_kappa
        elif typ == "xy" or typ == "yx":
            return self._get_Cl_kappa(ellmax)
        if typ == "yy":
            N0_kappa = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
            return self._get_Cl_kappa(ellmax) + 3*N0_kappa
        if typ == "zz":
            N0_kappa = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
            return self._get_Cl_kappa(ellmax) + 3*N0_kappa
        elif typ == "xz" or typ == "zx":
            return self._get_Cl_kappa(ellmax)
        elif typ == "zy" or typ == "yz":
            return self._get_Cl_kappa(ellmax)
        elif typ == "ww":
            N0_omega = self.noise.get_N0("curl", ellmax, ell_factors=self.N0_ell_factors)
            return N0_omega

    def _get_Covs(self, typ, Lmax, all_splines=False, nu=857e9, gal_win_zmin=None, gal_win_zmax=None, include_N0_kappa="both"):
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
            C1 = self._get_Cov(typ1 + typ1, Lmax, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
            C2 = self._get_Cov(typ2 + typ2, Lmax, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

    def _get_C_inv(self, typs, Lmax, nu, gal_win_zmin, gal_win_zmax):
        Ntyps = np.size(typs)
        all_typs = np.char.array(np.concatenate((typs, list('w'))))
        C = all_typs[:, None] + all_typs[None, :]
        C[:, Ntyps] = 0
        C[Ntyps, :] = 0
        C[Ntyps, Ntyps] = 'ww'
        args = C.flatten()
        args = args[args != '0']
        C_sym = Matrix(C)
        C_inv = C_sym.inv()
        C_inv_func = lambdify(args, C_inv)
        Covs = [self._get_Cov(arg, Lmax, nu, gal_win_zmin, gal_win_zmax) for arg in args]
        return C_inv_func(*Covs)

    def _get_optimal_Ns_sympy(self, Lmax, typ, typs, C_inv):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, tidy=True, ell_factors=self.N0_ell_factors))
        cov3_spline = N0_omega_spline

        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        return cov_inv1, cov_inv2, cov3_spline

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ, Lmin, nu, gal_win_zmin, gal_win_zmax, typs=None, C_inv=None, include_N0_kappa="both"):
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
            C1, C2, C3_spline = self._get_Covs(typ, Lmax, all_splines=False, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)
            if typ[0] != typ[1]:
                Cl = self._get_Cl(typ[:2], Lmax, nu, gal_win_zmin, gal_win_zmax)
                denom = ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl[None, Ls, None] * Cl[None, None, Ls])) * C3_spline(L3)
            else:
                denom = 2 * C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)
        return Ls, L3, dTheta, w, denom

    def _get_bispectrum_Fisher_vec(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa, Lmin, nu, gal_win_zmin, gal_win_zmax):
        Ls, L3, dTheta, w, denom = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin, nu, gal_win_zmin, gal_win_zmax, include_N0_kappa=include_N0_kappa)
        bispectrum = self.bi.get_bispectrum(typ, Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bispectrum ** 2 / denom)
        return f_sky/np.pi * I/((2*np.pi)**2)

    def _integral_prep_arr(self, Lmax, dL, Ntheta, typ, Lmin, nu, gal_win_zmin, gal_win_zmax, typs=None, C_inv=None, include_N0_kappa="both"):
        if typ[:3] == "opt":
            C1, C2, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv)
        else:
            C1, C2, C3_spline = self._get_Covs(typ, Lmax, all_splines=False, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(Lmin, Lmax + 1, dL)
        weights = np.ones(np.shape(thetas))
        return Ls, thetas, dTheta, weights, C1, C2, C3_spline

    # def _get_bispectrum_Fisher_arr_xx(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa, Lmin, nu, gal_win_zmin, gal_win_zmax):
    #     Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, typ, Lmin, nu, gal_win_zmin, gal_win_zmax, include_N0_kappa=include_N0_kappa)
    #     I = np.zeros(np.size(Ls))
    #     for iii, L1 in enumerate(Ls):
    #         I_tmp = 0
    #         for L2 in Ls[iii:]:
    #             L3 = self._get_L3(L1, L2, thetas)
    #             w = copy.deepcopy(weights)
    #             w[L3 > Lmax] = 0
    #             w[L3 < Lmin] = 0
    #             bispectrum = self.bi.get_bispectrum(typ, L1, L2, L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
    #             denom = 2 * C1[L1] * C2[L2] * C3_spline(L3)
    #             I_tmp += L2 * dL * 2 * np.dot(w, dTheta * bispectrum**2 / denom)
    #         I[iii] = 2 * np.pi * L1 * dL * I_tmp
    #     I *= 2 * f_sky/np.pi * 1/((2*np.pi)**2)
    #     return Ls, I

    def _get_bispectrum_Fisher_arr(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa, Lmin, nu, gal_win_zmin, gal_win_zmax):
        Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, typ, Lmin, nu, gal_win_zmin, gal_win_zmax, include_N0_kappa=include_N0_kappa)
        Cl_xy = self._get_Cl(typ[:2], Lmax, nu, gal_win_zmin, gal_win_zmax)
        I = np.zeros(np.size(Ls))
        for iii, L3 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls:
                L1 = self._get_third_L(L3, L2, thetas)
                w = copy.deepcopy(weights)
                w[L1 > Lmax] = 0
                w[L1 < Lmin] = 0
                bispectrum = self.bi.get_bispectrum(typ, L1, L2, L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
                if typ[0] == typ[1]:
                    denom = 2 * C1[L1] * C2[L2] * C3_spline(L3)
                else:
                    denom = ((C1[L1] * C2[L2]) + (Cl_xy[L1] * Cl_xy[L2])) * C3_spline(L3)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * bispectrum ** 2 / denom)
            I[iii] = 2 * np.pi * L3 * dL * I_tmp
        I *= f_sky / np.pi * 1 / ((2 * np.pi) ** 2)
        return Ls, I

    def _integral_prep_sample(self, Ls, Ntheta, typ, nu, gal_win_zmin, gal_win_zmax, typs=None, C_inv=None, include_N0_kappa="both"):
        Lmax = int(np.max(Ls))
        Lmin = int(np.min(Ls))
        if typ[:3] == "opt":
            C1, C2, C3_spline = self._get_optimal_Ns_sympy(Lmax, typ[4:], typs, C_inv)
        else:
            C1_spline, C2_spline, C3_spline = self._get_Covs(typ, Lmax, all_splines=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[:-1] = Ls[1:] - Ls[0:-1]
        dLs[-1] = dLs[-2]
        return Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    # def _get_bispectrum_Fisher_sample_xx(self, typ, Ls, Ntheta, f_sky, arr, include_N0_kappa, nu, gal_win_zmin, gal_win_zmax):
    #     Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta, typ, nu, gal_win_zmin, gal_win_zmax, include_N0_kappa=include_N0_kappa)
    #     I = np.zeros(np.size(Ls))
    #     for iii, L1 in enumerate(Ls):
    #         I_tmp = 0
    #         for jjj, L2 in enumerate(Ls[iii:]):
    #             L3 = self._get_L3(L1, L2, thetas)
    #             w = copy.deepcopy(weights)
    #             w[L3 > Lmax] = 0
    #             w[L3 < Lmin] = 0
    #             bispectrum = self.bi.get_bispectrum(typ, L1, L2, theta=thetas, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
    #             denom = 2 * C1_spline(L1) * C2_spline(L2) * C3_spline(L3)
    #             I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, bispectrum ** 2 / denom)
    #         I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
    #     I *= 2 * f_sky/np.pi * 1/((2 * np.pi) ** 2)
    #     if arr:
    #         return Ls, I
    #     return np.sum(I)

    def _get_bispectrum_Fisher_sample(self, typ, Ls, Ntheta, f_sky, arr, include_N0_kappa, nu, gal_win_zmin, gal_win_zmax):
        Lmax, Lmin, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta,typ, nu, gal_win_zmin, gal_win_zmax, include_N0_kappa=include_N0_kappa)
        I = np.zeros(np.size(Ls))
        Cl_xy_spline = self._interpolate(self._get_Cl(typ[:2], Lmax, nu, gal_win_zmin, gal_win_zmax))
        for iii, L3 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls):
                L1 = self._get_third_L(L3, L2, thetas)
                w = copy.deepcopy(weights)
                w[L1 > Lmax] = 0
                w[L1 < Lmin] = 0
                thetas12 = np.arcsin(L3 * np.sin(thetas) / L1)
                bispectrum = self.bi.get_bispectrum(typ, L1, L2, theta=thetas12, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
                if typ[0] == typ[1]:
                    denom = 2 * C1_spline(L1) * C2_spline(L2) * C3_spline(L3)
                else:
                    denom = ((C1_spline(L1) * C2_spline(L2)) + (Cl_xy_spline(L1) * Cl_xy_spline(L2))) * C3_spline(L3)
                I_tmp += L2 * dLs[jjj] * 2 * dTheta * np.dot(w, bispectrum ** 2 / denom)
            I[iii] = 2 * np.pi * L3 * dLs[iii] * I_tmp
        I *= f_sky/np.pi * 1/((2 * np.pi) ** 2)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_optimal_bispectrum_Fisher_element_vec(self, typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_win_zmin, gal_win_zmax):
        Ls, L3, dTheta, w, covs = self._integral_prep_vec(Lmax, dL, Ntheta, typ, Lmin=Lmin, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, typs=typs, C_inv=C_inv)
        bi1 = self.bi.get_bispectrum(typ[4:6]+"w", Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
        bi2 = self.bi.get_bispectrum(typ[6:] + "w", Ls[:, None], Ls[None, :], L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * bi1 * bi2 * covs)
        return 0.5 * f_sky/np.pi * I / ((2 * np.pi) ** 2)

    def _get_optimal_bispectrum_Fisher_element_arr(self, typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_win_zmin, gal_win_zmax):
        Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, typ, Lmin=Lmin, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, typs=typs, C_inv=C_inv)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls:
                L3 = self._get_third_L(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < Lmin] = 0
                bi1 = self.bi.get_bispectrum(typ[4:6] + "w", L1, L2, L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
                bi2 = self.bi.get_bispectrum(typ[6:] + "w", L1, L2, L3, M_spline=True, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax)
                covs = C1[L1] * C2[L2] / C3_spline(L3)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * bi1 * bi2 * covs)
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 0.5 * f_sky/np.pi * 1/((2 * np.pi) ** 2)
        return np.sum(I)

    def _get_optimal_bispectrum_Fisher(self, typs, Lmax, dL, Ntheta, f_sky, verbose, arr, nu, gal_win_zmin, gal_win_zmax):
        typs = np.char.array(typs)
        if 'I' in typs:
            Lmin = 110
        else:
            Lmin = 30     # 1808.07445 and https://cmb-s4.uchicago.edu/wiki/index.php/Survey_Performance_Expectations
        C_inv = self._get_C_inv(typs, Lmax, nu, gal_win_zmin, gal_win_zmax)
        if verbose:
            cond_10 = np.linalg.cond(C_inv)[:, :, 10]
            cond_100 = np.linalg.cond(C_inv)[:, :, 100]
            cond_1000 = np.linalg.cond(C_inv)[:, :, 1000]
            print(f"C_inv condition number at L = 10, 100, 1000: {cond_10:.2f}, {cond_100:.2f}, {cond_1000:.2f}")
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        F = 0
        perms = 0
        for iii in np.arange(Ncombos):
            for jjj in np.arange(Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                if arr:
                    F_tmp = self._get_optimal_bispectrum_Fisher_element_arr(typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_win_zmin, gal_win_zmax)
                else:
                    F_tmp = self._get_optimal_bispectrum_Fisher_element_vec(typs, typ, Lmax, dL, Ntheta, f_sky, C_inv, Lmin, nu, gal_win_zmin, gal_win_zmax)
                factor = 1
                perms += factor
                F += factor * F_tmp
                if verbose:
                    print(f"type = {typ}")
                    print(f"F = {F_tmp}")
                    print(f"count = {perms}")
        if perms != np.size(typs)**4:
            raise ValueError(f"{perms} permutations computed, should be {np.size(typs)**4}")
        return F

    def get_bispectrum_Fisher(self, typ, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False, Lmin=30, nu=857e9, gal_win_zmin=None, gal_win_zmax=None, include_N0_kappa="both"):
        """

        Parameters
        ----------
        typ
        Lmax
        dL
        Ls
        Ntheta
        f_sky
        arr
        Lmin
        nu
        gal_win_zmin
        gal_win_zmax
        include_N0_kappa

        Returns
        -------

        """
        self.bi.check_type(typ)
        if Ls is not None:
            return self._get_bispectrum_Fisher_sample(typ, Ls, Ntheta, f_sky, arr, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)
        if arr:
            return self._get_bispectrum_Fisher_arr(typ, Lmax, dL, Ntheta, f_sky, Lmin=Lmin, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)
        return self._get_bispectrum_Fisher_vec(typ, Lmax, dL, Ntheta, f_sky, Lmin=Lmin, nu=nu, gal_win_zmin=gal_win_zmin, gal_win_zmax=gal_win_zmax, include_N0_kappa=include_N0_kappa)


    def get_optimal_bispectrum_Fisher(self, typs="kg", Lmax=4000, dL=1, Ntheta=10, f_sky=1, verbose=False, arr=False, nu=857e9, gal_win_zmin=None, gal_win_zmax=None):
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
        return self._get_optimal_bispectrum_Fisher(typs, Lmax, dL, Ntheta, f_sky, verbose, arr, nu, gal_win_zmin, gal_win_zmax)

    def _get_ell_prim_prim(self, ell, ell_prim, theta):
        ell_prim_prim = self._maths.cosine_rule(ell, ell_prim, theta)
        theta_prim = self._maths.sine_rule(ell_prim_prim, theta, b=ell)
        return theta_prim, ell_prim_prim

    def _get_postborn_omega_ps(self, ells, ell_file, M_file, ell_prim_max, Nell_prim, Ntheta):
        mode = Modecoupling()
        ells_sample = np.load(ell_file)
        M = np.load(M_file)
        M_spline = mode.spline(ells_sample, M)
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi, dTheta, dtype=float)
        Lmax = max(ell_prim_max, 2 * ells[-1])
        dL = Lmax / Nell_prim
        Lprims = np.arange(2, Lmax + 1, dL)
        ells = ells[:, None]
        thetas = thetas[None, :]
        I = 0
        for Lprim in Lprims:
            theta_prims, Lprimprims = self._get_ell_prim_prim(ells, Lprim, thetas)
            I_tmp = 2 * Lprim * dL * dTheta * self._maths.cross(ells, Lprim, thetas) ** 2 * self._maths.dot(Lprim, Lprimprims,theta_prims) ** 2 / (Lprim ** 4 * Lprimprims ** 4) * M_spline.ev(Lprim, Lprimprims)
            I += I_tmp.sum(axis=1)
        return 4 * I / ((2 * np.pi) ** 2)

    def get_postborn_omega_ps(self, ells, ell_file, M_file, ell_prim_max=8000, Nell_prim=2000, Ntheta=100):
        """

        Parameters
        ----------
        ells
        ell_file
        M_file
        ell_prim_max
        Nell_prim
        Ntheta

        Returns
        -------

        """
        return self._get_postborn_omega_ps(ells, ell_file, M_file, ell_prim_max, Nell_prim, Ntheta)

    def get_rotation_ps_Fisher(self, Lmax, ell_file=None, M_file=None, f_sky=1, auto=True, camb=False):
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
            Cl = self.get_postborn_omega_ps(ells, ell_file, M_file)
        Cl_spline = InterpolatedUnivariateSpline(ells, Cl)
        ells = np.arange(2, Lmax + 1)
        N0 = self.noise.get_N0("curl", Lmax, True, self.N0_ell_factors)
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
