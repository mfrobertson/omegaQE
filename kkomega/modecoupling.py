from powerspectra import Powerspectra
from cosmology import Cosmology
import numpy as np


class Modecoupling:
    """
    Calculates the mode-coupling matrix.

    Attributes
    ----------
    weyl_PK : object
        CAMB RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self):
        """
        Constructor

        """
        self._cosmo = Cosmology()
        self._powerspectra = Powerspectra()
        self.weyl_PK = self._powerspectra.weyl_PK

    def _vectorise_ells(self, ells, ndim):
        if np.size(ells) == 1:
            return ells
        if ndim == 1:
            return ells[:, None]
        if ndim == 2:
            return ells[:, None, None]

    def _vectorise_zs(self, zs, Nells):
        ndim = zs.ndim
        if ndim == 1:
            return np.repeat(zs[np.newaxis, :], Nells, 0)
        if ndim == 2:
            return np.repeat(zs[np.newaxis, :, :], Nells, 0)

    def _integral_prep(self, Nchi):
        Chi_max = self._cosmo.get_chi_star()
        Chis = np.linspace(0, Chi_max, Nchi)
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        zs = zs[1:]
        Chis = Chis[1:]
        window = self._cosmo.window(Chis, Chi_max)
        return zs, Chis, dChi, window

    def _components(self, ells1, ells2, Nchi, kmin, kmax, extended, recalc_weyl):
        zs, Chis, dChi, win = self._integral_prep(Nchi)
        # zs = np.repeat(zs[np.newaxis, :], np.size(ells1), 0)
        Nells1 = np.size(ells1)
        ells1_vec = self._vectorise_ells(ells1, zs.ndim)
        zs = self._vectorise_zs(zs, Nells1)
        Cl_kappa = self._powerspectra.get_kappa_ps_2source(ells2, Chis, self._cosmo.get_chi_star(),
                                                           recalc_weyl=recalc_weyl)
        if recalc_weyl:
            self.weyl_PK = self._powerspectra.weyl_PK
        if extended:
            # ks = (ells1 +0.5)[:,None]*(1/Chis)
            ks = (ells1_vec + 0.5) / Chis
        else:
            ks = ells1_vec / Chis
        step = self._cosmo.heaviside(ks, kmin, kmax)
        weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=False, scaled=False)
        I = step * weyl_ps / Chis ** 2 * dChi * win ** 2 * Cl_kappa
        if extended:
            return I.sum(axis=1) * (ells1 + 0.5) ** 4
        return I.sum(axis=1) * ells1 ** 4

    def _matrix(self, ellmax, Nell, Nchi=100, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        ells2 = np.linspace(1, ellmax + 1, Nell)
        M = np.ones((Nell, Nell))
        for iii, ell1 in enumerate(ells2):
            M[iii, :] = self._components(np.ones(Nell)*ell1, ells2, Nchi, kmin, kmax, extended, recalc_weyl)
        return ells2, M

    def components(self, ells1, ells2, Nchi=100, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        """
        Performs the calculation for extracting components of the mode-coupling matrix.

        Parameters
        ----------
        ells1 : int or float or ndarray
            The multipole moment(s). Essentially the first indices of the matrix.
        ells2 : int or float or ndarray
            The multipole moment(s) with same dimensions as ell1. Essentially the second indices of the matrix.
        Nchi : int
            The number of steps in the integral during the calculation.
        kmin : int or float
            Minimum wavelength to be used in the calculation.
        kmax : int or float
            Minimum wavelength to be used in the calculation.
        extended : bool
            Use extended Limber approximation.
        recalc_weyl : bool
            Recalculate the Weyl potential power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        ndarray
            1D array of the matrix components at [ells, ells2].
        """
        return self._components(ells1, ells2, Nchi, kmin, kmax, extended, recalc_weyl)

    def spline(self, ellmax, Nell, Nchi=100, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        """
        Produces 2D spline of the mode coupling matrix.

        Parameters
        ----------
        ellmax : int
            Maximim moment.
        Nell : int
            Number of sample moments to use in the interpolation in each direction. Nell * Nell samples in total for 2D interpolation.
        Nchi : int
            The number of steps in the integral during the calculation.
        kmin : int or float
            Minimum wavelength to be used in the calculation.
        kmax : int or float
            Minimum wavelength to be used in the calculation.
        extended : bool
            Use extended Limber approximation.
        recalc_weyl : bool
            Recalculate the Weyl potential power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        RectBivariateSpline
            Returns the resulting spline object of the mode coupling matrix. RectBivariateSpline(ells1,ells2) will produce matrix. RectBivariateSpline.ev(ells1,ells2) will calculate components of the matrix.
        """
        from scipy.interpolate import RectBivariateSpline
        ells, M = self._matrix(ellmax, Nell, Nchi, kmin, kmax, extended, recalc_weyl)
        return RectBivariateSpline(ells, ells, M)