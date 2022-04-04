import numpy as np
from bispectra import Bispectra
from powerspectra import Powerspectra
from noise import Noise
from maths import Maths
from modecoupling import Modecoupling
from scipy.interpolate import InterpolatedUnivariateSpline
from cache.tools import getFileSep, path_exists
import copy

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

    def _replace_bad_Ls(self, N0):
        bad_Ls = np.where(N0 <= 0.)[0]
        for L in bad_Ls:
            if L > 1:
                N0[L] = 0.5 * (N0[L-1] + N0[L+1])
        return N0

    def _get_Cl_kappa(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_kappa_ps(ells)

    def _get_Cl_gal(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_gal_ps(ells)

    def _get_Cl_cib(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_ps(ells)

    def _get_Cl_gal_kappa(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_gal_kappa_ps(ells)

    def _get_Cl_cib_kappa(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_kappa_ps(ells)

    def _get_Cl_cib_gal(self,ellmax):
        ells = np.arange(ellmax + 1)
        return self.power.get_cib_kappa_ps(ells)

    def _get_Cl(self, typ, ellmax):
        if typ == "kk":
            N0_kappa = self.noise.get_N0("phi", ellmax, tidy=True, ell_factors=self.N0_ell_factors)
            return self._get_Cl_kappa(ellmax) + N0_kappa
        elif typ == "gk" or typ == "kg":
            return self._get_Cl_gal_kappa(ellmax)
        elif typ == "gg":
            N0_gal = self.noise.get_gal_shot_N(ellmax=ellmax)
            return self._get_Cl_gal(ellmax) + N0_gal
        elif typ == "Ik" or typ == "kI":
            return self._get_Cl_cib_kappa(ellmax)
        elif typ == "II":
            N0_gal = 0
            return self._get_Cl_cib(ellmax) + N0_gal
        elif typ == "Ig" or typ == "gI":
            return self._get_Cl_cib_gal(ellmax)


    def _get_L3(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _interpolate(self, arr):
        ells_sample = np.arange(np.size(arr))
        return InterpolatedUnivariateSpline(ells_sample, arr)

    def _get_cmb_Ns(self, Lmax, typ, include_N0_kappa, all_splines=False):
        N0_kappa = self.noise.get_N0("phi", Lmax, tidy=True, ell_factors=self.N0_ell_factors)
        Cl_kappa = self._get_Cl_kappa(Lmax)
        if include_N0_kappa == "both":
            C1 = Cl_kappa + N0_kappa
            C2 = Cl_kappa + N0_kappa
        elif include_N0_kappa == "one":
            C1 = Cl_kappa + (0.5 * N0_kappa)
            C2 = Cl_kappa
        else:
            C1 = Cl_kappa
            C2 = Cl_kappa
        if typ == "conv_rot":
            N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
            if all_splines:
                C1_spline = self._interpolate(C1)
                C2_spline = self._interpolate(C2)
                return C1_spline, C2_spline, N0_omega_spline
            return C1, C2, N0_omega_spline
        C3 = Cl_kappa + N0_kappa
        C3_spline = self._interpolate(C3)
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

    def _get_gal_Ns(self, Lmax, typ, all_splines=False):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
        C3_spline = N0_omega_spline
        Cl_gal = self._get_Cl_gal(Lmax)
        N_gal = self.noise.get_gal_shot_N(ellmax=Lmax)
        if typ == "gal_rot":
            C1 = Cl_gal + N_gal
            C2 = C1
        elif typ == "gal_conv_rot":
            Cl_kappa = self._get_Cl_kappa(Lmax)
            N0_kappa = self.noise.get_N0("phi", Lmax, tidy=True, ell_factors=self.N0_ell_factors)
            C1 = Cl_gal + N_gal
            C2 = Cl_kappa + N0_kappa
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

    def _get_cib_Ns(self, Lmax, typ, all_splines=False):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
        C3_spline = N0_omega_spline
        Cl_cib = self._get_Cl_cib(Lmax)
        N_cib = 0
        if typ == "cib_rot":
            C1 = Cl_cib + N_cib
            C2 = C1
        elif typ == "cib_conv_rot":
            Cl_kappa = self._get_Cl_kappa(Lmax)
            N0_kappa = self.noise.get_N0("phi", Lmax, tidy=True, ell_factors=self.N0_ell_factors)
            C1 = Cl_cib + N_cib
            C2 = Cl_kappa + N0_kappa
        elif typ == "cib_gal_rot":
            Cl_gal = self._get_Cl_gal(Lmax)
            N0_gal = self.noise.get_gal_shot_N(ellmax=Lmax)
            C1 = Cl_cib + N_cib
            C2 = Cl_gal + N0_gal
        if all_splines:
            C1_spline = self._interpolate(C1)
            C2_spline = self._interpolate(C2)
            return C1_spline, C2_spline, C3_spline
        return C1, C2, C3_spline

    def _get_optimal_Ns(self, Lmax, typ):
        N0_omega_spline = self._interpolate(self.noise.get_N0("curl", Lmax, ell_factors=self.N0_ell_factors))
        C3_spline = N0_omega_spline
        C1 = self._get_Cl(typ[0]+typ[2], Lmax)
        C2 = self._get_Cl(typ[1]+typ[3], Lmax)
        return C1, C2, C3_spline

    def _get_thetas(self, Ntheta):
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        return thetas, dTheta

    def _integral_prep_vec(self, Lmax, dL, Ntheta, typ, include_N0_kappa="both"):
        if typ == "conv" or typ == "conv_rot":
            C1, C2, C3_spline = self._get_cmb_Ns(Lmax, typ, include_N0_kappa)
        elif typ == "gal_rot" or typ == "gal_conv_rot":
            C1, C2, C3_spline = self._get_gal_Ns(Lmax, typ)
        elif typ == "cib_rot" or typ == "cib_conv_rot" or typ == "cib_gal_rot":
            C1, C2, C3_spline = self._get_cib_Ns(Lmax, typ)
        elif typ[:3] == "opt":
            C1, C2, C3_spline = self._get_optimal_Ns(Lmax, typ[4:])
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        L3 = self._get_L3(Ls[:, None], Ls[None, :], thetas[:, None, None])
        w = np.ones(np.shape(L3))
        w[L3 < 2] = 0
        w[L3 > Lmax] = 0
        return Ls, L3, dTheta, w, C1, C2, C3_spline

    def _get_bi(self, typ):
        if typ == "conv_rot" or typ == "kkw":
            return self.bi.get_convergence_rotation_bispectrum
        if typ == "conv":
            return self.bi.get_convergence_bispectrum
        elif typ == "gal_rot" or typ == "ggw":
            return self.bi.get_gal_rotation_bispectrum
        elif typ == "gal_conv_rot" or typ == "gkw" or typ == "kgw":
            return self.bi.get_gal_convergence_rotation_bispectrum
        elif typ == "cib_rot" or typ == "IIw":
            return self.bi.get_cib_rotation_bispectrum
        elif typ == "cib_conv_rot" or typ == "Ikw" or typ == "kIw":
            return self.bi.get_cib_convergence_rotation_bispectrum
        elif typ == "cib_gal_rot" or typ == "Igw" or typ == "gIw":
            return self.bi.get_cib_gal_rotation_bispectrum
        elif typ[:3] == "opt":
            bi1 = self._get_bi(typ[4:6]+"w")
            bi2 = self._get_bi(typ[6:]+"w")
            return bi1, bi2

    def _get_factor(self, typ, method):
        if typ == "conv_rot":
            if method == "vec":
                return 1
            elif method == "arr":
                return 2
        if typ == "conv":
            if method == "vec":
                return 1/3
            elif method == "arr":
                return 2/3
        elif typ == "gal_rot":
            if method == "vec":
                return 1
            elif method == "arr":
                return 2
        elif typ == "gal_conv_rot":
            if method == "vec":
                return 2
            elif method == "arr":
                return 2
        elif typ == "cib_rot":
            if method == "vec":
                return 1
            elif method == "arr":
                return 2
        elif typ == "cib_conv_rot":
            if method == "vec":
                return 2
            elif method == "arr":
                return 2
        elif typ == "cib_gal_rot":
            if method == "vec":
                return 2
            elif method == "arr":
                return 2

    def _get_denom(self, typ, C1, C2, C3, Ls, L3, Lmax):
        if typ == "conv_rot" or typ == "conv" or typ == "gal_rot" or typ == "cib_rot":
            return C1[None, Ls, None] * C2[None, None, Ls] * C3(L3)
        elif typ == "gal_conv_rot":
            Cl_gal_kappa = self._get_Cl_gal_kappa(Lmax)
            return ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl_gal_kappa[None, Ls, None] * Cl_gal_kappa[None, None, Ls])) * C3(L3)
        elif typ == "cib_conv_rot":
            Cl_cib_kappa = self._get_Cl_cib_kappa(Lmax)
            return ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl_cib_kappa[None, Ls, None] * Cl_cib_kappa[None, None, Ls])) * C3(L3)
        elif typ == "cib_gal_rot":
            Cl_cib_gal = self._get_Cl_cib_gal(Lmax)
            return ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl_cib_gal[None, Ls, None] * Cl_cib_gal[None, None, Ls])) * C3(L3)

    def _get_bispectrum_Fisher_vec(self, typ, Lmax, dL, Ntheta, f_sky, include_N0_kappa):
        Ls, L3, dTheta, w, C1, C2, C3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, typ, include_N0_kappa)
        bi = self._get_bi(typ)
        denom = self._get_denom(typ, C1, C2, C3_spline, Ls, L3, Lmax)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi(Ls[:, None], Ls[None, :], L3, M_spline=True) ** 2) / denom)
        factor = self._get_factor(typ, method="vec")
        return factor * I * f_sky / ((2 * np.pi) ** 3)

    def _get_optimal_bispectrum_Fisher_element_vec(self, typ, Lmax, dL, Ntheta, f_sky):
        Ls, L3, dTheta, w, C1, C2, C3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, typ)
        print(typ)
        bi1, bi2 = self._get_bi(typ)
        denom = C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)
        I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi1(Ls[:, None], Ls[None, :], L3, M_spline=True) * bi2(Ls[:, None], Ls[None, :], L3, M_spline=True) ) / denom)
        return I * f_sky / ((2 * np.pi) ** 3)

    def _get_optimal_bispectrum_Fisher(self, typs, Lmax, dL, Ntheta, f_sky):
        typs = np.char.array(typs)
        all_combos = typs[:, None] + typs[None, :]
        upper_matrix_flat = np.triu(all_combos).flatten()
        combos = upper_matrix_flat[upper_matrix_flat != '']
        Ncombos = np.size(combos)
        F = 0
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = "opt_" + combos[iii] + combos[jjj]
                F_tmp = self._get_optimal_bispectrum_Fisher_element_vec(typ, Lmax, dL, Ntheta, f_sky)
                print(F_tmp)
                if combos[iii] == combos[jjj]:
                    factor = 1
                else:
                    factor = np.size(typs)
                F += factor * F_tmp
        return F

    # def _get_convergence_rotation_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky, include_N0_kappa):
    #     Ls, L3, dTheta, w, C1, C2, N3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv_rot", include_N0_kappa)
    #     bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
    #     I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_rot_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * N3_spline(L3)))
    #     return I * f_sky / ((2 * np.pi) ** 3)
    #
    # def _get_convergence_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky, include_N0_kappa):
    #     Ls, L3, dTheta, w, C1, C2, C3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "conv", include_N0_kappa)
    #     bi_conv = self.bi.get_convergence_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
    #     I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_conv ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * C3_spline(L3)))
    #     return I * f_sky / (3 * (2*np.pi) ** 3)
    #
    # def _get_gal_rotation_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky):
    #     Ls, L3, dTheta, w, C1, C2, N3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "gal_rot")
    #     bi_gal_rot = self.bi.get_gal_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
    #     I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_gal_rot ** 2)/ (C1[None, Ls, None] * C2[None, None, Ls] * N3_spline(L3)))
    #     return I * f_sky / ((2 * np.pi) ** 3)
    #
    # def _get_gal_convergence_rotation_bispectrum_Fisher_vec(self, Lmax, dL, Ntheta, f_sky):
    #     Ls, L3, dTheta, w, C1, C2, N3_spline = self._integral_prep_vec(Lmax, dL, Ntheta, "gal_conv_rot")
    #     bi_gal_conv_rot = self.bi.get_gal_convergence_rotation_bispectrum(Ls[:,None], Ls[None,:], L3, M_spline=True)
    #     Cl_gal_kappa = self._get_Cl_gal_kappa(Lmax)
    #     denom = ((C1[None, Ls, None] * C2[None, None, Ls]) + (Cl_gal_kappa[None, Ls, None] * Cl_gal_kappa[None, None, Ls])) * N3_spline(L3)
    #     I = 2 * 2 * np.pi * dL * dL * np.sum(Ls[None, :, None] * Ls[None, None, :] * dTheta * w * (bi_gal_conv_rot ** 2)/ denom)
    #     return 2 * I * f_sky / ((2 * np.pi) ** 3)

    def _integral_prep_sample(self, Ls, Ntheta, typ, include_N0_kappa="both"):
        Lmax = int(Ls[-1])
        if typ == "conv" or typ == "conv_rot":
            C1_spline, C2_spline, C3_spline = self._get_cmb_Ns(Lmax, typ, include_N0_kappa, all_splines=True)
        elif typ == "gal_rot" or typ == "gal_conv_rot":
            C1_spline, C2_spline, C3_spline = self._get_gal_Ns(Lmax, typ, all_splines=True)
        elif typ == "cib_rot" or typ == "cib_conv_rot" or typ == "cib_gal_rot":
            C1_spline, C2_spline, C3_spline = self._get_cib_Ns(Lmax, typ, all_splines=True)
        thetas, dTheta = self._get_thetas(Ntheta)
        weights = np.ones(np.size(thetas))
        dLs = np.ones(np.size(Ls))
        dLs[0] = Ls[0]
        dLs[1:] = Ls[1:] - Ls[0:-1]
        return Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline

    def _get_convergence_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr, include_N0_kappa):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "conv_rot", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_rot_conv = self.bi.get_convergence_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_rot_conv ** 2) / (C1_spline(L1) * C2_spline(L2) * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_convergence_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr, include_N0_kappa):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, C3_spline = self._integral_prep_sample(Ls, Ntheta, "conv", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_conv ** 2) / (C1_spline(L1) * C2_spline(L2) * C3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/(3 * (2*np.pi) ** 3)
        if arr:
            return I
        return np.sum(I)

    def _get_gal_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "gal_rot")
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_gal_rot = self.bi.get_gal_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_gal_rot ** 2) / (C1_spline(L1) * C2_spline(L2) * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_gal_convergence_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "gal_conv_rot")
        Cl_gal_kappa_spline = self._interpolate(self._get_Cl_gal_kappa(Lmax))
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_gal_conv_rot = self.bi.get_gal_convergence_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                denom = ((C1_spline(L1) * C2_spline(L2)) + (Cl_gal_kappa_spline(L1) * Cl_gal_kappa_spline(L2))) * N3_spline(L3)
                I_tmp += L2 * dLs[jjj] * 2 * dTheta * np.dot(w, (bi_gal_conv_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_cib_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "cib_rot")
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls[iii:]):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_rot = self.bi.get_cib_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                I_tmp += L2 * dLs[iii + jjj] * 2 * dTheta * np.dot(w, (bi_cib_rot ** 2) / (C1_spline(L1) * C2_spline(L2) * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_cib_convergence_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "cib_conv_rot")
        Cl_cib_kappa_spline = self._interpolate(self._get_Cl_cib_kappa(Lmax))
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_conv_rot = self.bi.get_cib_convergence_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                denom = ((C1_spline(L1) * C2_spline(L2)) + (Cl_cib_kappa_spline(L1) * Cl_cib_kappa_spline(L2))) * N3_spline(L3)
                I_tmp += L2 * dLs[jjj] * 2 * dTheta * np.dot(w, (bi_cib_conv_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _get_cib_gal_rotation_bispectrum_Fisher_sample(self, Ls, Ntheta, f_sky, arr):
        Lmax, dLs, thetas, dTheta, weights, C1_spline, C2_spline, N3_spline = self._integral_prep_sample(Ls, Ntheta, "cib_gal_rot")
        Cl_cib_gal_spline = self._interpolate(self._get_Cl_cib_gal(Lmax))
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for jjj, L2 in enumerate(Ls):
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_gal_rot = self.bi.get_cib_gal_rotation_bispectrum(L1, L2, theta=thetas, M_spline=True)
                denom = ((C1_spline(L1) * C2_spline(L2)) + (Cl_cib_gal_spline(L1) * Cl_cib_gal_spline(L2))) * N3_spline(L3)
                I_tmp += L2 * dLs[jjj] * 2 * dTheta * np.dot(w, (bi_cib_gal_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dLs[iii] * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return I
        return np.sum(I)

    def _integral_prep_arr(self, Lmax, dL, Ntheta, typ, include_N0_kappa="both"):
        if typ == "conv" or typ == "conv_rot":
            C1, C2, C3_spline = self._get_cmb_Ns(Lmax, typ, include_N0_kappa)
        elif typ == "gal_rot" or typ == "gal_conv_rot":
            C1, C2, C3_spline = self._get_gal_Ns(Lmax, typ)
        elif typ == "cib_rot" or typ == "cib_conv_rot" or typ == "cib_gal_rot":
            C1, C2, C3_spline = self._get_cib_Ns(Lmax, typ)
        thetas, dTheta = self._get_thetas(Ntheta)
        Ls = np.arange(2, Lmax + 1, dL)
        weights = np.ones(np.shape(thetas))
        return Ls, thetas, dTheta, weights, C1, C2, C3_spline

    # def _get_bispectrum_Fisher_arr(self, typ, Lmax, dL, Ntheta, f_sky, arr, include_N_kappa="both"):
    #     Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, typ, include_N_kappa)
    #     Cl_gal_kappa = self._get_Cl_gal_kappa(Lmax)
    #     Cl_cib_kappa = self._get_Cl_cib_kappa(Lmax)
    #     I = np.zeros(np.size(Ls))
    #     factor = self._get_factor(typ, method="arr")
    #     print(factor)
    #     bi = self._get_bi(typ)
    #     print(bi)
    #     for iii, L1 in enumerate(Ls):
    #         I_tmp = 0
    #         for L2 in Ls:
    #             L3 = self._get_L3(L1, L2, thetas)
    #             w = copy.deepcopy(weights)
    #             w[L3 > Lmax] = 0
    #             w[L3 < 2] = 0
    #             if typ == "gal_conv_rot":
    #                 denom = ((C1[L1] * C2[L2]) + (Cl_gal_kappa[L1] * Cl_gal_kappa[L2])) * C3_spline(L3)
    #             elif typ == "cib_conv_rot":
    #                 denom = ((C1[L1] * C2[L2]) + (Cl_cib_kappa[L1] * Cl_cib_kappa[L2])) * C3_spline(L3)
    #             else:
    #                 denom = (C1[L1] * C2[L2] * C3_spline(L3))
    #             I_tmp += L2 * dL * 2 * np.dot(w, dTheta * bi(L1, L2, L3, M_spline=True) ** 2 / denom)
    #         I[iii] = 2 * np.pi * L1 * dL * I_tmp
    #     I *= factor * f_sky / ((2 * np.pi) ** 3)
    #     if arr:
    #         return Ls, I
    #     return np.sum(I)

    def _get_convergence_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv_rot", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv_rot = self.bi.get_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_conv_rot ** 2) / (C1[L1] * C2[L2] * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2*f_sky/((2*np.pi)**3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_convergence_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa):
        Ls, thetas, dTheta, weights, C1, C2, C3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "conv", include_N0_kappa)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_conv = self.bi.get_convergence_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_conv ** 2) / (C1[L1] * C2[L2] * C3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= f_sky / (12 * np.pi ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_gal_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "gal_rot")
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_gal_rot = self.bi.get_gal_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_gal_rot ** 2) / (C1[L1] * C2[L2] * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2 * f_sky / ((2 * np.pi) ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_gal_convergence_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "gal_conv_rot")
        Cl_gal_kappa = self._get_Cl_gal_kappa(Lmax)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_gal_conv_rot = self.bi.get_gal_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                denom = ((C1[L1] * C2[L2]) + (Cl_gal_kappa[L1] * Cl_gal_kappa[L2])) * N3_spline(L3)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_gal_conv_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2 * f_sky / ((2 * np.pi) ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_cib_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "cib_rot")
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls[iii:]:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                if L1 == L2:
                    w[:] = 0.5
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_rot = self.bi.get_cib_rotation_bispectrum(L1, L2, L3, M_spline=True)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_cib_rot ** 2) / (C1[L1] * C2[L2] * N3_spline(L3)))
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2 * f_sky / ((2 * np.pi) ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_cib_convergence_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "cib_conv_rot")
        Cl_cib_kappa = self._get_Cl_cib_kappa(Lmax)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_conv_rot = self.bi.get_cib_convergence_rotation_bispectrum(L1, L2, L3, M_spline=True)
                denom = ((C1[L1] * C2[L2]) + (Cl_cib_kappa[L1] * Cl_cib_kappa[L2])) * N3_spline(L3)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_cib_conv_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2 * f_sky / ((2 * np.pi) ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def _get_cib_gal_rotation_bispectrum_Fisher_arr(self, Lmax, dL, Ntheta, f_sky, arr):
        Ls, thetas, dTheta, weights, C1, C2, N3_spline = self._integral_prep_arr(Lmax, dL, Ntheta, "cib_gal_rot")
        Cl_cib_gal = self._get_Cl_cib_gal(Lmax)
        I = np.zeros(np.size(Ls))
        for iii, L1 in enumerate(Ls):
            I_tmp = 0
            for L2 in Ls:
                L3 = self._get_L3(L1, L2, thetas)
                w = copy.deepcopy(weights)
                w[L3 > Lmax] = 0
                w[L3 < 2] = 0
                bi_cib_gal_rot = self.bi.get_cib_gal_rotation_bispectrum(L1, L2, L3, M_spline=True)
                denom = ((C1[L1] * C2[L2]) + (Cl_cib_gal[L1] * Cl_cib_gal[L2])) * N3_spline(L3)
                I_tmp += L2 * dL * 2 * np.dot(w, dTheta * (bi_cib_gal_rot ** 2) / denom)
            I[iii] = 2 * np.pi * L1 * dL * I_tmp
        I *= 2 * f_sky / ((2 * np.pi) ** 3)
        if arr:
            return Ls, I
        return np.sum(I)

    def get_convergence_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False, include_N0_kappa="both"):
        """
        Computes the Fisher information for the kappa kappa omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.
        include_N0_kappa : bool
            'both' = keep both convegence noise terms, 'one' = keep only one convergnce noise term, 'none' = no convergence noise terms.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_convergence_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa)
            else:
                #return self._get_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky, include_N0_kappa)
                return self._get_bispectrum_Fisher_vec("conv_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa)
        return self._get_convergence_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr, include_N0_kappa)


    def get_convergence_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False, include_N0_kappa="both"):
        """
        Computes the Fisher information for the leading order post born kappa kappa kappa bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.
        include_N0_kappa : bool
            'both' = keep both convegence noise terms, 'one' = keep only one convergnce noise term, 'none' = no convergence noise terms.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_convergence_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr, include_N0_kappa)
            else:
                #return self._get_convergence_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky, include_N0_kappa)
                return self._get_bispectrum_Fisher_vec("conv", Lmax, dL, Ntheta, f_sky, include_N0_kappa)
        return self._get_convergence_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr, include_N0_kappa)

    def get_gal_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
        """
        Computes the Fisher information for the leading order post born g g omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_gal_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                #return self._get_gal_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
                return self._get_bispectrum_Fisher_vec("gal_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa=None)
        return self._get_gal_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)

    def get_gal_convergence_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
        """
        Computes the Fisher information for the leading order post born g kappa omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_gal_convergence_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                #return self._get_gal_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
                return self._get_bispectrum_Fisher_vec("gal_conv_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa=None)
        return self._get_gal_convergence_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)

    def get_cib_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
        """
        Computes the Fisher information for the leading order post born I I omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_cib_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                #return self._get_gal_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
                return self._get_bispectrum_Fisher_vec("cib_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa=None)
        return self._get_cib_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)

    def get_cib_convergence_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
        """
        Computes the Fisher information for the leading order post born I kappa omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_cib_convergence_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                #return self._get_gal_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
                return self._get_bispectrum_Fisher_vec("cib_conv_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa=None)
        return self._get_cib_convergence_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)

    def get_cib_gal_rotation_bispectrum_Fisher(self, Lmax=4000, dL=1, Ls=None, Ntheta=10, f_sky=1, arr=False):
        """
        Computes the Fisher information for the leading order post born I g omega bispectrum.

        Parameters
        ----------
        Lmax : int
            Maximum multipole moment limit in the integrals.
        dL : int
            Step size of the integrals.
        Ls : ndarray
            Alternative to supplying Lmax and dL; 1D array of the sampled multipole moments wished to be integrated over.
        Ntheta : int
            Number of steps to use in the angular integral.
        f_sky : int or float
            Fraction of sky.
        arr : bool
            Return an array of Fisher value for each step in the first integral.

        Returns
        -------
            float or 2-tuple
        If arr = False then the Fisher information is returned as float. If True then a 1D array of multipole moment steps are returned as the first part of the tuple, the second is the corresponding Fisher values at each moment.
        """
        if Ls is None:
            if arr:
                return self._get_cib_gal_rotation_bispectrum_Fisher_arr(Lmax, dL, Ntheta, f_sky, arr)
            else:
                #return self._get_gal_convergence_rotation_bispectrum_Fisher_vec(Lmax, dL, Ntheta, f_sky)
                return self._get_bispectrum_Fisher_vec("cib_gal_rot", Lmax, dL, Ntheta, f_sky, include_N0_kappa=None)
        return self._get_cib_gal_rotation_bispectrum_Fisher_sample(Ls, Ntheta, f_sky, arr)

    def get_optimal_bispectrum_Fisher(self, typs="kg", Lmax=4000, dL=1, Ntheta=10, f_sky=1):
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
        return self._get_optimal_bispectrum_Fisher(typs, Lmax, dL, Ntheta, f_sky)

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
