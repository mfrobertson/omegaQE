import camb
import omegaqe
from camb import postborn
import numpy as np
from scipy.constants import Planck, physical_constants
import omegaqe.tools as tools
from omegaqe.tools import maths
from pathlib import Path
import os
from scipy.interpolate import InterpolatedUnivariateSpline


class Cosmology:
    """
    Container for useful cosmological functionality. All CAMB functionality is initialised with parameters from Lensit.
    """

    def __init__(self, paramfile=omegaqe.CAMB_FILE):
        """
        Constructor.
        """

        self.paramfile = paramfile
        self.set_cosmology(self.paramfile)
        self.cib_norms = None
        self.dn_dz_splines = None
        self.dn_dz_tot_spline = None
        self.gal_biases = None
        self.agora = False
        self.b1 = 0.84
        self.s_spline = self.get_s_spline(1)

    def update_gal_bias(self, b1):
        self.b1 = b1

    def set_cosmology(self, paramfile=None):
        self._pars = self.get_params(paramfile)
        self._results = self.calc_results()

    def calc_results(self):
        return camb.get_results(self._pars)

    def get_params(self, paramfile=None):
        paramfile = self.paramfile if paramfile is None else paramfile
        return self._get_pars(self._get_param_file(paramfile))

    def get_pars_dict(self, pars: camb.model.CAMBparams):
        thetastar = 0.010410837983195698
        mnu = 0.06
        sig8 = 0.8123981609602227
        b1 = 0.84
        return {"thetastar": thetastar,
                "H0": pars.H0,
                "100thetastar": 100*thetastar,
                "ombh2": pars.ombh2,
                "omch2": pars.omch2,
                "lnombh2": np.log(pars.ombh2),
                "lnomch2": np.log(pars.omch2),
                "omk": pars.omk,
                "mnu": mnu,
                "tau": pars.Reion.optical_depth,
                "w": pars.DarkEnergy.w,
                "wa": pars.DarkEnergy.wa,
                "As": pars.InitPower.As,
                "ns": pars.InitPower.ns,
                "sig8": sig8,
                "b1": b1,
                }

    def modify_params(self, pars, mod_dict, H0=False):
        if H0:
            self._pars = camb.set_params(cp=pars, H0=mod_dict["H0"], ombh2=mod_dict["ombh2"], omch2=mod_dict["omch2"],
                        omk=mod_dict["omk"], mnu=mod_dict["mnu"], tau=mod_dict["tau"], nnu=3.046,
                        standard_neutrino_neff=3.046, w=mod_dict["w"], wa=mod_dict["wa"], As=mod_dict["As"],
                        ns=mod_dict["ns"])
        else:
            self._pars = camb.set_params(cp=pars, thetastar=mod_dict["thetastar"], ombh2=mod_dict["ombh2"], omch2=mod_dict["omch2"],
                        omk=mod_dict["omk"], mnu=mod_dict["mnu"], tau=mod_dict["tau"], nnu=3.046,
                        standard_neutrino_neff=3.046, w=mod_dict["w"], wa=mod_dict["wa"], As=mod_dict["As"],
                        ns=mod_dict["ns"], theta_H0_range=(1, 1000))
        self.update_gal_bias(mod_dict["b1"])
        self._results = self.calc_results()

    def _get_param_file(self, name):
        if name.lower() == "lensit":
            return "Lensit_fiducial_flatsky_params.ini"
        if name.lower() == "demnunii":
            return "DEMNUnii_params.ini"
        if name.lower() == "agora":
            return "AGORA_params.ini"
        if name.lower() == "planck":
            return "Planck_2018.ini"
        return name

    def _get_pars(self, filename):
        if filename is None:
            print("Setting up cosmology with CAMB default parameters")
            return camb.CAMBparams()
        sep = tools.getFileSep()
        path = rf"{omegaqe.DATA_DIR}{sep}CAMB{sep}{filename}"
        if os.path.isfile(path):
            print(f"Setting up cosmology with CAMB ini file at {path}")
            return camb.read_ini(path)
        raise ValueError(f"CAMB parameter input file {path} does not exist.")

    def cmb_lens_window(self, Chi1, Chi2, heaviside=True):
        """
        Computes the Window function for CMB lensing.

        Parameters
        ----------
        Chi1 : int or float or ndarray
            Comoving radial distance [Mpc]. Usually the one being integrated over.
        Chi2 : int or float
            Comiving radial distance [Mpc]. Usually the limit, and Chi2 > Chi1.
        heaviside : bool
            Perform a Heaviside step where Chi1 > Chi2.

        Returns
        -------
        int or float or ndarray
            Returns the computed Window function. The dimensions will be equivalent to Chi1.
        """
        win = (Chi2 - Chi1) / (Chi1 * Chi2)
        if np.size(Chi1) == 1 and heaviside and Chi1 > Chi2:
            return 0
        if heaviside:
            if np.size(Chi1) == 1:
                win = np.array([win])
            return maths.heaviside_steps(win) * win
        return win

    def cmb_lens_window_matter(self, Chi1, Chi2, heaviside=True):
        """
        Computes the Window function for CMB lensing when using matter power spectrum instead of Weyl spectrum.

        Parameters
        ----------
        Chi1 : int or float or ndarray
            Comoving radial distance [Mpc]. Usually the one being integrated over.
        Chi2 : int or float
            Comiving radial distance [Mpc]. Usually the limit, and Chi2 > Chi1.
        heaviside : bool
            Perform a Heaviside step where Chi1 > Chi2.

        Returns
        -------
        int or float or ndarray
            Returns the computed Window function. The dimensions will be equivalent to Chi1.
        """
        zs = self.Chi_to_z(Chi1)
        poisson_fac = self.poisson_factor(zs)
        win = self.cmb_lens_window(Chi1, Chi2, heaviside) * poisson_fac
        return win

    def gal_lens_window(self, Chi1, Chi2, gal_distro="LSST_gold", heaviside=True):
        """
        Reference 1411.0115

        Parameters
        ----------
        Chi1
        Chi2
        heaviside

        Returns
        -------

        """
        chis = np.linspace(Chi1, Chi2, 1000)
        dChi = chis[1] - chis[0]

        # Setting bias to 1 in gal window
        gal_distro = self.gal_window_Chi(chis, typ=gal_distro, bias_unity=True)

        I = gal_distro * self.cmb_lens_window(Chi1, chis, heaviside)
        q = np.sum(dChi * I, axis=0)

        return q

    def gal_lens_window_matter(self, Chi1, Chi2, gal_distro="LSST_gold", heaviside=True):
        """
        Reference 1411.0115

        Parameters
        ----------
        Chi1
        Chi2
        heaviside

        Returns
        -------

        """
        zs = self.Chi_to_z(Chi1)
        poisson_fac = self.poisson_factor(zs)
        win = self.gal_lens_window(Chi1, Chi2, gal_distro, heaviside) * poisson_fac
        return win
    
    def mu_window(self, Chi1, Chi2, gal_distro="LSST_gold", heaviside=True):
        """
        Reference 1411.0115

        Parameters
        ----------
        Chi1
        Chi2
        heaviside

        Returns
        -------

        """
        s = self.s_spline(self.Chi_to_z(Chi1))
        fac = (5*s - 2)
        return fac * self.gal_lens_window(Chi1, Chi2, gal_distro, heaviside)
    
    def mu_window_matter(self, Chi1, Chi2, gal_distro="LSST_gold", heaviside=True):
        """
        Reference 1411.0115

        Parameters
        ----------
        Chi1
        Chi2
        heaviside

        Returns
        -------

        """
        s = self.s_spline(self.Chi_to_z(Chi1))
        fac = (5*s - 2)
        return fac * self.gal_lens_window_matter(Chi1, Chi2, gal_distro, heaviside)

    def rsd_window_constChi(self, Chi, ellmax, typ="LSST_gold", zmin=0, zmax=None):
        """
        Reference 2309.00052

        Parameters
        ----------
        Chi1
        Chi2
        heaviside

        Returns
        -------

        """
        ells = np.arange(9, ellmax + 1)
        L0 = ((2 * ells ** 2) + (2 * ells) - 1) / ((2 * ells - 1) * (2 * ells + 3))
        L_m1 = - ((ells * (ells - 1)) / ((2 * ells - 1) * np.sqrt((2 * ells - 3) * (2 * ells + 1))))
        L_p1 = - (((ells + 1) * (ells + 2)) / ((2 * ells + 3) * np.sqrt((2 * ells + 1) * (2 * ells + 5))))
        L_facs = np.array([L_m1, L0, L_p1])
        res = np.zeros(np.size(ells))
        for iii, L_fac in enumerate(L_facs):
            iii -= 1
            Chi_scaled = (2 * ells + 1 + 4 * iii) / (2 * ells + 1) * Chi
            n = self.gal_window_Chi(Chi_scaled, typ=typ, zmin=zmin, zmax=zmax, bias_unity=True)
            f = self.get_f(self.Chi_to_z(Chi_scaled))
            res += L_fac * n * f
        return np.concatenate((np.zeros(9), res))

    def poisson_factor(self, z):
        return (1 + z) * self.z_to_Chi(z) ** 2 * 3 / 2 * self._pars.omegam * self.get_hubble(0) ** 2

    def setup_dndz_splines(self, zs, dn_dzs, biases):
        self.dn_dz_splines = np.empty(np.shape(dn_dzs)[0], dtype=InterpolatedUnivariateSpline)
        self.gal_biases = np.empty(np.shape(dn_dzs)[0])
        dn_dz_tot = None
        for iii, dn_dz in enumerate(dn_dzs):
            self.dn_dz_splines[iii] = InterpolatedUnivariateSpline(zs, dn_dz, ext=1)
            if dn_dz_tot is None:
                dn_dz_tot = dn_dz
            else:
                dn_dz_tot += dn_dz
            self.gal_biases[iii] = biases[iii]
        self.dn_dz_tot_spline = InterpolatedUnivariateSpline(zs, dn_dz_tot, ext=1)

    def _gal_z_LSST_distribution(self, z):
        # 1705.02332 equation 14 and B1
        z0 = 0.311
        return 1 / (2 * z0) * (z / z0) ** 2 * np.exp(-z / z0)

    def _agora_dist(self, z):
        # z0 = 0.13
        # alpha = 0.78
        # return z**2*np.exp(-z/z0)**alpha

        # zs = np.linspace(0,1200,10000)
        # n_tot = None
        # for iii in np.arange(5):
        #     if n_tot is None:
        #         n_tot = self.dn_dz_splines[iii](zs)
        #     else:
        #         n_tot += self.dn_dz_splines[iii](zs)
        # return InterpolatedUnivariateSpline(zs, n_tot)(z)
        return self.dn_dz_tot_spline(z)

    def _gal_z_CMB_distribution(self, z):
        return self.cmb_lens_window(self.z_to_Chi(z), self.get_chi_star()) / self.get_hubble(z)

    def _gal_z_flat_distribution(self, z):
        Chi_distro = np.ones(np.shape(z))
        z_str = self.Chi_to_z(self.get_chi_star())
        Chi_distro[z > z_str] = 0
        Chi_distro[z < 0] = 0
        return Chi_distro / self.get_hubble(z)

    def _check_z_distr_typ(self, typ):
        typs = ["LSST_gold", "LSST_gold_bias_unity", "CMB", "flat", "flat_bias_unity", "perfect"]
        if typ not in typs:
            raise ValueError(f"Redshift distribution type {typ} not from accepted types: {typs}")

    def _get_z_distr_func(self, typ):
        # self._check_z_distr_typ(typ)
        if typ == "LSST_gold":
            return self._gal_z_LSST_distribution
        if typ == "LSST_a":
            return self.dn_dz_splines[0]
        if typ == "LSST_b":
            return self.dn_dz_splines[1]
        if typ == "LSST_c":
            return self.dn_dz_splines[2]
        if typ == "LSST_d":
            return self.dn_dz_splines[3]
        if typ == "LSST_e":
            return self.dn_dz_splines[4]
        if typ == "CMB":
            return self._gal_z_CMB_distribution
        if typ == "flat":
            return self._gal_z_flat_distribution
        if typ == "flat_bias_unity":
            return self._get_z_distr_func("flat")
        if typ == "perfect":
            return self._get_z_distr_func("flat")
        if typ == "agora":
            return self._agora_dist
        else:
            raise ValueError(f"No galaxy distribution of type {typ}.")

    def _get_bias(self, z, typ, bias_unity):
        if bias_unity:
            return 1
        if typ == "flat_bias_unity" or typ == "LSST_gold_bias_unity" or typ == "perfect":
            return 1
        if typ == "LSST_a":
            return self.gal_biases[0]
        if typ == "LSST_b":
            return self.gal_biases[1]
        if typ == "LSST_c":
            return self.gal_biases[2]
        if typ == "LSST_d":
            return self.gal_biases[3]
        if typ == "LSST_e":
            return self.gal_biases[4]
        if typ == "agora":
            if np.size(z) == 1:
                z = np.array([z])
            bias = 1 + (self.b1 * z)
            bias[np.logical_and(z > 0.2, z < 0.4)] = self.gal_biases[0]
            bias[np.logical_and(z > 0.4, z < 0.6)] = self.gal_biases[1]
            bias[np.logical_and(z > 0.6, z < 0.8)] = self.gal_biases[2]
            bias[np.logical_and(z > 0.8, z < 1.0)] = self.gal_biases[3]
            bias[np.logical_and(z > 1.0, z < 1.2)] = self.gal_biases[4]
            return bias
        return 1 + (self.b1 * z)
    
    def get_s_spline(self, typ=1):
        zs = np.linspace(0, 200, 10000)
        if typ == 1:
            s = np.exp(zs) / (np.exp(1))
        elif typ == 2:
            s = np.exp(zs) / (np.exp(2))
        elif typ == 3:
            s = np.exp(zs) / (np.exp(3))
        elif typ == 4:
            s = np.ones(np.size(zs))
        return InterpolatedUnivariateSpline(zs, s)

    def gal_window_z(self, z, typ="LSST_gold", zmin=None, zmax=None, bias_unity=False):
        """
        1705.02332 equation 14 and B1 (originally from 0912.0201)
        Parameters
        ----------
        z

        Returns
        -------

        """
        z_distr_func = self._get_z_distr_func(typ)
        dn_dz = z_distr_func(z)
        b = self._get_bias(z, typ, bias_unity)
        zmin = 0 if zmin is None else zmin
        zmax = self.Chi_to_z(self.get_chi_star()) if zmax is None else zmax
        zs = np.linspace(zmin, zmax, 10000)
        dz = zs[1] - zs[0]
        norm = 1 if typ == "perfect" else np.sum(dz * z_distr_func(zs))
        window = (dn_dz * b) / norm
        return maths.rectangular_pulse_steps(z, zmin, zmax) * window

    def gal_window_Chi(self, Chi, typ="LSST_gold", zmin=None, zmax=None, bias_unity=False):
        """
        1906.08760 eq 2.7
        Parameters
        ----------
        Chi
        heaviside
        Chi_edge1
        Chi_edge2

        Returns
        -------

        """
        z = self.Chi_to_z(Chi)
        window_z = self.gal_window_z(z, typ, zmin, zmax, bias_unity)
        window = window_z * self.get_hubble(z)
        return window

    def _gal_window_z_no_norm(self, z, typ="LSST_gold", zmin=None, zmax=None, bias_unity=False):
        z_distr_func = self._get_z_distr_func(typ)
        dn_dz = z_distr_func(z)
        b = 1 if bias_unity else 1 + (self.b1 * z)
        zs = np.linspace(0, self.Chi_to_z(self.get_chi_star()), 4000)
        dz = zs[1] - zs[0]
        norm = np.sum(dz * z_distr_func(zs))
        window = (dn_dz * b) / norm
        if zmin is not None and zmax is not None:
            return maths.rectangular_pulse_steps(z, zmin, zmax) * window
        return window

    def gal_window_fraction(self, zmin, zmax, typ="LSST_gold"):
        """

        Parameters
        ----------
        zmin
        zmax

        Returns
        -------

        """
        zs = np.linspace(zmin, zmax, 4000)
        window = self._gal_window_z_no_norm(zs, typ=typ, zmin=zmin, zmax=zmax)
        total_zs = np.linspace(0, self.Chi_to_z(self.get_chi_star()), 4000)
        total_window = self._gal_window_z_no_norm(total_zs, typ=typ)
        dz = zs[1] - zs[0]
        I = np.sum(dz * window)
        dz = total_zs[1] - total_zs[0]
        I_total = np.sum(dz * total_window)
        return I / I_total

    def _SED_func(self, nu):
        # "1801.05396 uses 857 GHz (pg. 2)"
        # "1705.02332 uses 353 GHz (pg. 4)"
        nu_prim = 4955e9  # according to 1705.02332 this is in 1502.01591 (also in  Planck lensing 2015)
        alpha = 2  # 0912.4315 and Planck 2015 appendix D
        beta = 2
        power = beta + 3
        T = 34
        h = Planck
        k_B = physical_constants["Boltzmann constant"][0]
        exponent = np.asarray((h * nu) / (k_B * T), dtype=np.double)
        exponent[exponent > 700] = np.double(700)
        small_nu = (np.exp(exponent) - 1) ** -1 * nu ** power
        big_nu = (np.exp(exponent) - 1) ** -1 * nu_prim ** power * (nu / nu_prim) ** -alpha
        if np.shape(nu) != ():
            w1 = np.zeros(np.shape(nu))
            w2 = np.zeros(np.shape(nu))
            w1[nu <= nu_prim] = 1
            w2[nu > nu_prim] = 1
            return w1 * small_nu + w2 * big_nu
        if nu <= nu_prim:
            return small_nu
        return big_nu

    def _get_cib_norm(self, nu):
        if self.cib_norms is None:
            self.cib_norms = np.load(Path(__file__).parent / "data/planck_cib/b_c.npy")
        if nu == 353e9:
            if self.agora: return 6.48e-65 * 1e-6  # My fit of AGORA cib between ell of 110 and 2000
            return 5.28654e-65 * 1e-6  # From Toshiya, matching 1705.02332 and 2110.09730
            # return 7.75714689e-65* 1e-6
            # return self.cib_norms[0]

            # return 8.24989321e-71
            # b_c = 8.71253313e-65 * 1e-6   # 1e-6 to change units of window to MJy/sr
        if nu == 545e9:
            return self.cib_norms[1]
            # return 7.52485062e-71
            # b_c = 8.76989271e-65 * 1e-6
        if nu == 857e9:
            return self.cib_norms[2]
            # return 5.71686654e-71
            # b_c = 7.68698899e-65 * 1e-6

    def _cib_window_z_sSED(self, z, nu, b_c=None):
        # "1801.05396 uses 857 GHz (pg. 2)"
        # "1705.02332 uses 353 GHz (pg. 4)"
        """
        1705.02332 equation 12 (originally from 0912.4315)
        Parameters
        ----------
        z

        Returns
        -------

        """
        if b_c is None:
            b_c = self._get_cib_norm(nu)
        Chi = self.z_to_Chi(z)
        H = self.get_hubble(z)
        z_c = 2
        sig_z = 2
        window = (Chi ** 2) / (H * (1 + z) ** 2) * np.exp(-((z - z_c) ** 2) / (2 * sig_z ** 2)) * self._SED_func(
            nu * (z + 1))
        return b_c * window

    def _cib_window_Chi_sSED(self, Chi, nu=353e9, b_c=None):
        # "1801.05396 uses 857 GHz (pg. 2)"
        # "1705.02332 uses 353 GHz (pg. 4)"
        z = self.Chi_to_z(Chi)
        return self._cib_window_z_sSED(z, nu, b_c) * self.get_hubble(z)

    def cib_window_Chi(self, Chi, nu=353e9, b_c=None):
        return self._cib_window_Chi_sSED(Chi, nu, b_c)

    def cib_window_z(self, z, nu=353e9, b_c=None):
        return self._cib_window_z_sSED(z, nu, b_c)

    def get_chi_star(self):
        """

        Returns
        -------
        float
            The comoving radial distance [Mpc] of the surface of last scattering.
        """
        return self.get_eta0() - self._results.tau_maxvis

    def get_eta0(self):
        """

        Returns
        -------
        float
            The present day conformal time [Mpc].
        """
        return self._results.conformal_time(0)

    def _hubble_2dim(self, z):
        H = np.zeros(np.shape(z))
        for iii in range(np.shape(H)[0]):
            H[iii] = self.get_hubble(z[iii])
        return H

    def _hubble_3dim(self, z):
        H = np.zeros(np.shape(z))
        for iii in range(np.shape(H)[0]):
            H[iii] = self._hubble_2dim(z[iii])
        return H

    def get_hubble(self, z):
        if np.shape(z) != ():
            if z.ndim == 2:
                return self._hubble_2dim(z)
            elif z.ndim == 3:
                return self._hubble_3dim(z)
        return self._results.h_of_z(z)

    def _eta_to_z_2dim(self, eta):
        z = np.zeros(np.shape(eta))
        for iii in range(np.shape(z)[0]):
            z[iii] = self.eta_to_z(eta[iii])
        return z

    def eta_to_z(self, eta):
        """

        Parameters
        ----------
        eta : int or float or ndarray
            Conformal time [Mpc].

        Returns
        -------
        float or ndarray
            The redhsift at the corresponding conformal time(s).
        """
        if np.shape(eta) != ():
            if eta.ndim == 2:
                return self._eta_to_z_2dim(eta)
        return self._results.redshift_at_conformal_time(eta)

    def _Chi_to_z_2dim(self, Chi):
        z = np.zeros(np.shape(Chi))
        for iii in range(np.shape(z)[0]):
            z[iii] = self.Chi_to_z(Chi[iii])
        return z

    def _Chi_to_z_3dim(self, Chi):
        z = np.zeros(np.shape(Chi))
        for iii in range(np.shape(z)[0]):
            z[iii] = self._Chi_to_z_2dim(Chi[iii])
        return z

    def Chi_to_z(self, Chi):
        """

        Parameters
        ----------
        Chi : int or float or ndarray
            Comoving radial distance [Mpc].

        Returns
        -------
        float or ndarray
            The redhsift at the corresponding Chi(s).
        """
        if np.shape(Chi) != ():
            if Chi.ndim == 2:
                return self._Chi_to_z_2dim(Chi)
            elif Chi.ndim == 3:
                return self._Chi_to_z_3dim(Chi)
        return self._results.redshift_at_comoving_radial_distance(Chi)

    def _z_to_Chi_2dim(self, z):
        Chi = np.zeros(np.shape(z))
        for iii in range(np.shape(Chi)[0]):
            Chi[iii] = self.z_to_Chi(z[iii])
        return Chi

    def _z_to_Chi_3dim(self, z):
        Chi = np.zeros(np.shape(z))
        for iii in range(np.shape(Chi)[0]):
            Chi[iii] = self._z_to_Chi_2dim(z[iii])
        return Chi

    def z_to_Chi(self, z):
        """

        Parameters
        ----------
        z : int or float or ndarray
            Redshift.

        Returns
        -------
        float
            The comoving radial distance [Mpc] of the corresponding redshifts.
        """
        if np.shape(z) != ():
            if z.ndim == 2:
                return self._z_to_Chi_2dim(z)
            elif z.ndim == 3:
                return self._z_to_Chi_3dim(z)
        return self._results.comoving_radial_distance(z)

    def get_f(self, z):
        # Reference 1807.06209
        Om_c = self._results.get_Omega("cdm", z)
        Om_b = self._results.get_Omega("baryon", z)
        Om_nu_mass = self._results.get_Omega("nu", z)
        Om_nu = self._results.get_Omega("neutrino", z)
        return (Om_c + Om_b + Om_nu + Om_nu_mass) ** 0.55

    def _get_ps_variables(self, typ):
        weyl = "Weyl"
        matter = "delta_tot"
        cdm = "delta_cdm"
        if typ.lower() == "weyl":
            return weyl, weyl
        if typ.lower() == "matter":
            return matter, matter
        if typ.lower() == "weyl-matter" or typ.lower() == "matter-weyl":
            return matter, weyl
        if typ.lower() == "cdm":
            return cdm, cdm

    def get_matter_PK(self, kmax=None, zmax=None, typ="Weyl"):
        """
        Gets an interpolated matter power spectrum from CAMB.

        Parameters
        ----------
        kmax : int or float
            Maximum wavenumber [Mpc] from which to extract the matter power spectrum interpolator.
        zmax : int or float
            Maximum redshift from which to extract the matter power spectrum interpolator.

        Returns
        -------
        object
            Returns a RectBivariateSpline PK object.
        """
        if zmax is None:
            zmax = self.eta_to_z(self.get_eta0() - self.get_chi_star()) + 100
        if kmax is None:
            kmax = 100
        var1, var2 = self._get_ps_variables(typ)
        PK = camb.get_matter_power_interpolator(self._pars, hubble_units=False, zmin=0, zmax=zmax, kmax=kmax,
                                                k_hunit=False, var1=var1, var2=var2)
        return PK

    def get_matter_ps(self, PK, z, k, curly=False, weyl_scaled=True, typ="weyl"):
        """
        Returns the matter power spectrum spectrum.

        Parameters
        ----------
        weylPK : object
            The matter power spectrum interpolator in the form of a RectBivariateSpline PK object.
        z : int or float or ndarray
            Redshift.
        k : int or float or ndarray
            [Mpc^-1].
        curly : bool
            Return dimensionless power spectrum.
        scaled : bool
            Accept default CAMB scaling of Weyl potential by k^2.

        Returns
        -------
        ndarray
            Weyl power spectrum calculated at specific points z and k.
        """
        ps = PK.P(z, k, grid=False)
        if not weyl_scaled:
            if typ.lower() == "weyl":
                ps *= k ** -4
            elif typ.lower() == "matter-weyl" or typ.lower() == "weyl-matter":
                ps *= k ** -2
        if curly:
            return ps * k ** 3 / (2 * np.pi ** 2)
        return ps

    def get_postborn_omega_ps(self, ellmax=20000, acc=1):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return postborn.get_field_rotation_power(self._pars, lmax=ellmax, acc=acc)

    def get_cmb_ps(self, ellmax=4000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        cmb_ps = self._results.get_cmb_power_spectra(self._pars, lmax=ellmax, spectra=['total'], CMB_unit="muK",
                                                     raw_cl=True)
        return cmb_ps['total'][:, 0]

    def get_grad_lens_ps(self, typ, ellmax=6000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return_zeros = False
        if typ == "TT":
            index = 0
        elif typ == "EE":
            index = 1
        elif typ == "BB":
            index = 2
        elif typ == "TE" or typ == "ET":
            index = 4
        elif typ == "TB" or typ == "BT":
            return_zeros = True
        elif typ == "EB" or typ == "BE":
            return_zeros = True
        else:
            raise ValueError(f"Type {typ} does not exist.")
        spectra = self._results.get_lensed_gradient_cls(ellmax, raw_cl=True)
        if return_zeros:
            return np.zeros(np.shape(spectra[:, 0]))
        return spectra[:, index]

    def get_lens_ps(self, typ, ellmax=6000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return_zeros = False
        if typ == "TT":
            index = 0
        elif typ == "EE":
            index = 1
        elif typ == "BB":
            index = 2
        elif typ == "TE" or typ == "ET":
            index = 3
        elif typ == "TB" or typ == "BT" or typ == "EB" or typ == "BE":
            return_zeros = True
        else:
            raise ValueError(f"Type {typ} does not exist.")
        spectra = self._results.get_lensed_scalar_cls(lmax=ellmax + 10, raw_cl=True)
        if return_zeros:
            return np.zeros(np.shape(spectra[:, 0]))
        return spectra[:, index]

    def get_unlens_ps(self, typ, ellmax=6000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return_zeros = False
        if typ == "TT":
            index = 0
        elif typ == "EE":
            index = 1
        elif typ == "BB":
            index = 2
        elif typ == "TE" or typ == "ET":
            index = 3
        elif typ == "TB" or typ == "BT" or typ == "EB" or typ == "BE":
            return_zeros = True
        else:
            raise ValueError(f"Type {typ} does not exist.")
        spectra = self._results.get_unlensed_scalar_cls(lmax=ellmax + 10, raw_cl=True)
        if return_zeros:
            return np.zeros(np.shape(spectra[:, 0]))
        return spectra[:, index]

