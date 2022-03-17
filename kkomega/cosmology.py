import camb
from camb import postborn
import numpy as np
import os
from cache import tools
from maths import Maths

class Cosmology:
    """
    Container for useful cosmological functionality. All CAMB functionality is initialised with parameters from Lensit.
    """

    def __init__(self):
        """
        Constructor.
        """
        self._maths = Maths()
        dir_current = os.path.dirname(os.path.realpath(__file__))
        sep = tools.getFileSep()
        self._pars = camb.read_ini(rf"{dir_current}{sep}data{sep}Lensit_fiducial_flatsky_params.ini")
        self._results = camb.get_background(self._pars)


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
            return self._maths.heaviside_steps(win) * win
        return win

    def _gal_z_distribution(self, z):
        z0 = 0.311
        return 1/(2*z0) * (z/z0)**2 * np.exp(-z/z0)


    def gal_cluster_window(self, Chi, heaviside=False, Chi_edge1=None, Chi_edge2=None):
        """
        1705.02332 equation 14 and B1
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
        z0 = 0.311
        n = self._gal_z_distribution(z)
        b = 1 + 0.84*z
        zs = np.linspace(0, 100, 2000)
        dz = zs[1] - zs[0]
        norm = np.sum(dz*self._gal_z_distribution(zs))
        window = (n * b)/norm
        if heaviside:
            return self._maths.rectangular_pulse_steps(Chi, Chi_edge1, Chi_edge2) * window
        return window

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
        return self._results.redshift_at_comoving_radial_distance(Chi)

    def _z_to_Chi_2dim(self, z):
        Chi = np.zeros(np.shape(z))
        for iii in range(np.shape(Chi)[0]):
            Chi[iii] = self.z_to_Chi(z[iii])
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

    def get_postborn_omega_ps(self, ellmax=20000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return postborn.get_field_rotation_power(self._pars, lmax=ellmax)
