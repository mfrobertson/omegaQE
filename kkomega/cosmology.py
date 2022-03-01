import camb
from camb import postborn
import numpy as np
import os
from cache import tools

class Cosmology:
    """
    Container for useful cosmological functionality. All CAMB functionality is initialised with parameters from Lensit.
    """

    def __init__(self):
        """
        Constructor.
        """
        dir_current = os.path.dirname(os.path.realpath(__file__))
        sep = tools.getFileSep()
        self._pars = camb.read_ini(rf"{dir_current}{sep}data{sep}Lensit_fiducial_flatsky_params.ini")
        self._results = camb.get_background(self._pars)


    def window(self, Chi1, Chi2, heaviside=True):
        """
        Computes the Window function.

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
            win[win < 0] = 0
        return win


    def rectangular_pulse_steps(self, arr, min, max):
        """
        Produces the steps of a rectangular pulse function for given boundaries acting on the input array.

        Parameters
        ----------
        arr : ndarray
            Array for which the steps will be calculated.
        min : int or float
            Minimum boundary.
        max : int or float
            Maximum boundary.

        Returns
        -------
        ndarray
            The steps of same dimensions as the input array, containing the steps resulting from a rectangular pulse given the boundaries.
        """
        step = np.ones(arr.shape)
        step[:] = 1
        step[arr < min] = 0
        step[arr > max] = 0
        return step


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
        return self._results.redshift_at_conformal_time(eta)

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
        return self._results.redshift_at_comoving_radial_distance(Chi)


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
        return self._results.comoving_radial_distance(z)


    def get_weyl_PK(self, kmax=None, zmax=None):
        """
        Gets an interpolated Weyl potential from CAMB.

        Parameters
        ----------
        kmax : int or float
            Maximum wavenumber [Mpc] from which to extract the Weyl potential interpolator.
        zmax : int or float
            Maximum redshift from which to extract the Weyl potential interpolator.

        Returns
        -------
        object
            Returns a RectBivariateSpline PK object.
        """
        if zmax is None:
            zmax = self.eta_to_z(self.get_eta0() - self.get_chi_star()) + 100
        if kmax is None:
            kmax = 100
        PK_weyl = camb.get_matter_power_interpolator(self._pars, hubble_units=False, zmin=0, zmax=zmax, kmax=kmax,
                                                     k_hunit=False, var1="Weyl", var2="Weyl")
        return PK_weyl


    def get_weyl_ps(self, weylPK, z, k, curly=False, scaled=True):
        """
        Returns the Weyl power spectrum.

        Parameters
        ----------
        weylPK : object
            The weyl interpolator in the form of a RectBivariateSpline PK object.
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
        ps = weylPK.P(z, k, grid=False)
        if not scaled:
            ps *= k**-4
        if curly:
            return ps * k** 3 / (2 * np.pi ** 2)
        return ps

    def get_omega_ps(self, ellmax=20000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return postborn.get_field_rotation_power(self._pars, lmax=ellmax)
