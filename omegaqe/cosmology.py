import camb
import omegaqe
from camb import postborn
import numpy as np
from scipy.constants import Planck, physical_constants
import omegaqe.tools as tools
from omegaqe.tools import maths
from pathlib import Path
import os

class Cosmology:
    """
    Container for useful cosmological functionality. All CAMB functionality is initialised with parameters from Lensit.
    """

    def __init__(self, paramfile=omegaqe.CAMB_FILE):
        """
        Constructor.
        """

        self._pars = self._get_pars(self._get_param_file(paramfile))
        self._results = camb.get_results(self._pars)
        self.cib_norms = None

    def _get_param_file(self, name):
        if name.lower() == "lensit":
            return "Lensit_fiducial_flatsky_params.ini"
        if name.lower() == "demnunii":
            return "DEMNUnii_params.ini"
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
            # win[win < 0] = 0
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

    def gal_lens_window(self, Chi1, Chi2, heaviside=True):
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
        gal_distro = self.gal_window_Chi(chis, bias_unity=True)

        I = gal_distro * self.cmb_lens_window(Chi1, chis, heaviside)
        q = np.sum(dChi * I, axis=0)

        return q

    def gal_lens_window_matter(self, Chi1, Chi2, heaviside=True):
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
        win = self.gal_lens_window(Chi1, Chi2, heaviside) * poisson_fac
        return win

    def poisson_factor(self, z):
        return (1 + z) * self.z_to_Chi(z) ** 2 * 3 / 2 * self._pars.omegam * self.get_hubble(0) ** 2

    def _gal_z_LSST_distribution(self, z):
        # 1705.02332 equation 14 and B1
        z0 = 0.311
        return 1/(2*z0) * (z/z0)**2 * np.exp(-z/z0)

    def _gal_z_CMB_distribution(self, z):
        return self.cmb_lens_window(self.z_to_Chi(z), self.get_chi_star())/self.get_hubble(z)

    def _gal_z_flat_distribution(self, z):
        Chi_distro = np.ones(np.shape(z))
        z_str = self.Chi_to_z(self.get_chi_star())
        Chi_distro[z>z_str] = 0
        Chi_distro[z<0] = 0
        return Chi_distro/self.get_hubble(z)

    def _check_z_distr_typ(self, typ):
        typs = ["LSST_gold", "LSST_gold_bias_unity", "CMB", "flat", "flat_bias_unity", "perfect"]
        if typ not in typs:
            raise ValueError(f"Redshift distribution type {typ} not from accepted types: {typs}")

    def _get_z_distr_func(self, typ):
        self._check_z_distr_typ(typ)
        if typ == "LSST_gold":
            return self._gal_z_LSST_distribution
        if typ == "CMB":
            return self._gal_z_CMB_distribution
        if typ == "flat":
            return self._gal_z_flat_distribution
        if typ == "flat_bias_unity":
            return self._get_z_distr_func("flat")
        if typ == "perfect":
            return self._get_z_distr_func("flat")
        else:
            raise ValueError(f"No galaxy distribution of type {typ}.")

    def _get_bias(self, z, typ, bias_unity):
        if bias_unity:
            return 1
        if typ == "flat_bias_unity" or typ == "LSST_gold_bias_unity" or typ == "perfect":
            return 1
        return 1 + (0.84*z)

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
        zs = np.linspace(0, self.Chi_to_z(self.get_chi_star()), 4000)    #TODO: optimize step size here
        dz = zs[1] - zs[0]
        if zmin is not None and zmax is not None:
            norm = 1 if typ == "perfect" else np.sum(dz * maths.rectangular_pulse_steps(zs, zmin, zmax) * z_distr_func(zs))
            window = (dn_dz * b) / norm
            return maths.rectangular_pulse_steps(z, zmin, zmax) * window
        norm = 1 if typ == "perfect" else np.sum(dz * z_distr_func(zs))
        window = (dn_dz * b) / norm
        return window

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
        b = 1 if bias_unity else 1 + (0.84 * z)
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
        #"1801.05396 uses 857 GHz (pg. 2)"
        #"1705.02332 uses 353 GHz (pg. 4)"
        nu_prim = 4955e9  # according to 1705.02332 this is in 1502.01591 (also in  Planck lensing 2015)
        alpha = 2       # 0912.4315 and Planck 2015 appendix D
        beta = 2
        power = beta + 3
        T = 34
        h = Planck
        k_B = physical_constants["Boltzmann constant"][0]
        exponent = np.asarray((h*nu)/(k_B*T), dtype=np.double)
        exponent[exponent>700] = np.double(700)
        small_nu = (np.exp(exponent) - 1) ** -1 * nu ** power
        big_nu = (np.exp(exponent) - 1)**-1 * nu_prim**power * (nu/nu_prim)**-alpha
        if np.shape(nu) != ():
            w1 = np.zeros(np.shape(nu))
            w2 = np.zeros(np.shape(nu))
            w1[nu<=nu_prim] = 1
            w2[nu>nu_prim] = 1
            return w1*small_nu + w2*big_nu
        if nu<=nu_prim:
            return small_nu
        return big_nu

    def _get_cib_norm(self, nu):
        if self.cib_norms is None:
            self.cib_norms = np.load(Path(__file__).parent/"data/planck_cib/b_c.npy")
        if nu == 353e9:
            return 5.28654e-65*1e-6  # From Toshiya, matching 1705.02332 and 2110.09730
            # return 7.75714689e-65* 1e-6
            # return self.cib_norms[0]

            #return 8.24989321e-71
            # b_c = 8.71253313e-65 * 1e-6   # 1e-6 to change units of window to MJy/sr
        if nu == 545e9:
            return self.cib_norms[1]
            #return 7.52485062e-71
            # b_c = 8.76989271e-65 * 1e-6
        if nu == 857e9:
            return self.cib_norms[2]
            #return 5.71686654e-71
            # b_c = 7.68698899e-65 * 1e-6

    def _cib_window_z_sSED(self, z, nu, b_c=None):
        #"1801.05396 uses 857 GHz (pg. 2)"
        #"1705.02332 uses 353 GHz (pg. 4)"
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
        window = (Chi ** 2) / (H * (1 + z) ** 2) * np.exp(-((z - z_c) ** 2) / (2 * sig_z ** 2)) * self._SED_func(nu*(z + 1))
        return b_c*window

    def _cib_window_Chi_sSED(self, Chi, nu=353e9, b_c=None):
        #"1801.05396 uses 857 GHz (pg. 2)"
        #"1705.02332 uses 353 GHz (pg. 4)"
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

    def _get_ps_variables(self, typ):
        weyl = "Weyl"
        matter = "delta_tot"
        if typ.lower() == "weyl":
            return weyl, weyl
        if typ.lower() == "matter":
            return matter, matter
        if typ.lower() == "weyl-matter" or typ.lower() == "matter-weyl":
            return matter, weyl

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
                ps *= k**-4
            elif typ.lower() == "matter-weyl" or typ.lower() == "weyl-matter":
                ps *= k**-2
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
        cmb_ps = self._results.get_cmb_power_spectra(self._pars, lmax=ellmax, spectra=['total'],CMB_unit="muK", raw_cl=True)
        return cmb_ps['total'][:,0]

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
            return np.zeros(np.shape(spectra[:,0]))
        return spectra[:,index]


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
            return np.zeros(np.shape(spectra[:,0]))
        return spectra[:,index]

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
            return np.zeros(np.shape(spectra[:,0]))
        return spectra[:,index]

